"""Collect a license-clean track-imagery pool from Wikimedia Commons.

Why Commons and not broadcast screenshots: the dataset has to be *published* to
the Hugging Face Hub. Broadcast frames are copyrighted and cannot be
redistributed; Commons motorsport photography is CC-BY / CC-BY-SA / public
domain, so the dataset ships with proper attribution and judges can download it.

The useful discovery is that Commons maintains a dedicated category tree,
"Formula One in rain in the <decade>s", which covers the wet-trackside
domain this project needs and which no Hub dataset covers.

Rate limits: Commons enforces a robot policy and will 429 aggressively. Requests
are therefore serial with a delay and exponential backoff - a parallel fetch
gets ~3% of images through and silently loses the rest. The category listing is
cached to data/raw/candidates.jsonl so a re-run costs no API calls.

Output:
    data/raw/pool/<sha1>.jpg     downloaded thumbnails
    data/raw/candidates.jsonl    cached category listing
    data/raw/manifest.jsonl      one record per downloaded image, with attribution

Usage:
    python -m data_pipeline.collect_commons discover
    python -m data_pipeline.collect_commons collect
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia enforces a User-Agent policy. Measured on
# 12 sequential thumbnail fetches at 1 req/s:
#
#   "WeatherWhiplash/1.0 (HF hackathon dataset)"            2/12   ten 429s
#   "WeatherWhiplash/1.0 (<contact>) python-urllib/3.9"    12/12   zero 429s
#   spoofed Chrome UA                                       9/12   three 429s
#
# The compliant form is both the polite option and the fastest one. Spoofing a
# browser is explicitly against their robot policy *and* performs worse.
# Set COMMONS_CONTACT to your own email or project URL so you are identifiable.
CONTACT = os.environ.get(
    "COMMONS_CONTACT", "https://huggingface.co/spaces; weather-whiplash hackathon project"
)
UA = {
    "User-Agent": f"WeatherWhiplash/1.0 ({CONTACT}) python-urllib/3.9",
    "Accept": "image/jpeg,image/png,image/*;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
}

ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / "data" / "raw" / "pool"
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
CANDIDATES = ROOT / "data" / "raw" / "candidates.jsonl"

THUMB_WIDTH = 768
MIN_WIDTH = 400
REQUEST_DELAY = 1.1  # between API calls
DOWNLOAD_DELAY = 0.35  # between image fetches

#: How many images to actually download per weak-label pool. Wet is uncapped
#: because Damp and Drying examples only exist in the rain categories and are
#: the scarce classes; ordinary dry race photography is abundant and subsampled.
HINT_CAPS: Dict[str, Optional[int]] = {"wet": None, "mixed": 450, "dry": 200}

# (category, weak label hint, cap). The hint is only a *prior* that seeds
# triage - every image is re-checked before it reaches the training set.
CATEGORY_PLAN: List[Tuple[str, str, int]] = [
    # --- the rain tree: highest-precision wet trackside imagery on Commons ---
    ("Category:Formula One in rain in the 2010s", "wet", 400),
    ("Category:Formula One in rain in the 2000s", "wet", 200),
    ("Category:Formula One in rain in the 2020s", "wet", 200),
    ("Category:Formula One in rain in the 1990s", "wet", 200),
    ("Category:Formula One in rain in the 1960s", "wet", 200),
    ("Category:Formula One in rain in the 1970s", "wet", 200),
    ("Category:Formula One in rain in the 1980s", "wet", 200),
    # --- known wet race weekends ---
    # The richest source of Damp and Drying frames: one weekend contains dry
    # practice, a wet race and the drying laps between, all at the same circuit
    # from the same camera positions. Framing and surface stay fixed while the
    # water varies, which a sample scattered across circuits does not give.
    ("Category:2011 Canadian Grand Prix", "mixed", 240),
    ("Category:2022 Monaco Grand Prix", "mixed", 180),
    ("Category:2009 Chinese Grand Prix", "mixed", 140),
    ("Category:2008 British Grand Prix", "mixed", 120),
    ("Category:2012 Malaysian Grand Prix", "mixed", 110),
    ("Category:2015 United States Grand Prix", "mixed", 60),
    ("Category:2010 Chinese Grand Prix", "mixed", 50),
    ("Category:2007 Japanese Grand Prix", "mixed", 40),
    ("Category:2008 Monaco Grand Prix", "mixed", 40),
    ("Category:2017 Italian Grand Prix", "mixed", 30),
    ("Category:2010 Korean Grand Prix", "mixed", 25),
    # --- circuits and dry weekends: the Dry class ---
    ("Category:Silverstone Circuit", "dry", 150),
    ("Category:Nürburgring", "dry", 140),
    ("Category:Circuit de Spa-Francorchamps", "dry", 115),
    ("Category:Hungaroring", "dry", 95),
    ("Category:Autodromo Nazionale Monza", "dry", 55),
    ("Category:Circuit Paul Ricard", "dry", 40),
    ("Category:Red Bull Ring", "dry", 35),
    ("Category:Bahrain International Circuit", "dry", 25),
    ("Category:Interlagos", "dry", 20),
    ("Category:2013 Indian Grand Prix", "dry", 180),
    ("Category:2019 Hungarian Grand Prix", "dry", 35),
    ("Category:2019 Japanese Grand Prix", "dry", 25),
    ("Category:2015 Bahrain Grand Prix", "dry", 12),
]


# ---------------------------------------------------------------------------
# MediaWiki client
# ---------------------------------------------------------------------------


def api_get(params: Dict[str, str], tries: int = 6) -> dict:
    """GET the Commons API with exponential backoff."""
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    last: Optional[Exception] = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            time.sleep(REQUEST_DELAY * (2**attempt) + random.random())
    raise RuntimeError(f"Commons API failed after {tries} tries: {last}")


def strip_html(value: Optional[str]) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def iter_category_files(category: str, limit: int) -> Iterator[dict]:
    """Yield file pages in a category, following continuation."""
    if not category.startswith("Category:"):
        category = "Category:" + category
    seen = 0
    cont: Optional[str] = None
    while seen < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|sha1|size|mime",
            "iiurlwidth": str(THUMB_WIDTH),
        }
        if cont:
            params["gcmcontinue"] = cont
        data = api_get(params)
        time.sleep(REQUEST_DELAY)
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            return
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("thumburl"):
                continue
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if int(info.get("width") or 0) < MIN_WIDTH:
                continue
            yield {"title": page.get("title"), "info": info}
            seen += 1
            if seen >= limit:
                return
        cont = (data.get("continue") or {}).get("gcmcontinue")
        if not cont:
            return


def record_from_page(page: dict, category: str, hint: str) -> dict:
    info = page["info"]
    meta = info.get("extmetadata") or {}

    def field(key: str) -> str:
        return strip_html((meta.get(key) or {}).get("value"))

    return {
        "sha1": info.get("sha1"),
        "title": page["title"],
        "thumburl": info["thumburl"],
        "descriptionurl": info.get("descriptionurl"),
        "license": field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
        "artist": field("Artist"),
        "credit": field("Credit"),
        "usage_terms": field("UsageTerms"),
        "category": category,
        "hint": hint,
        "width": info.get("width"),
        "height": info.get("height"),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> None:
    """Search the category namespace so the plan above can be extended."""
    terms = args.terms or [
        "Formula One in rain",
        "Grand Prix 2019",
        "motor racing circuit",
        "rallying rain",
    ]
    for term in terms:
        data = api_get(
            {"action": "query", "list": "search", "srsearch": term,
             "srnamespace": "14", "srlimit": "12"}
        )
        time.sleep(REQUEST_DELAY)
        print(f"\n=== {term}")
        for hit in (data.get("query") or {}).get("search", []):
            title = hit["title"]
            counts = api_get(
                {"action": "query", "list": "categorymembers", "cmtitle": title,
                 "cmtype": "file", "cmlimit": "500"}
            )
            time.sleep(REQUEST_DELAY)
            n = len((counts.get("query") or {}).get("categorymembers", []))
            if n >= args.min_files:
                print(f"    {n:4d} files  {title}")


def build_candidates() -> List[dict]:
    """List every planned category and cache the result."""
    found: Dict[str, dict] = {}
    for category, hint, cap in CATEGORY_PLAN:
        before = len(found)
        try:
            for page in iter_category_files(category, cap):
                rec = record_from_page(page, category, hint)
                if rec["sha1"] and rec["sha1"] not in found:
                    found[rec["sha1"]] = rec
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {category}: {exc}", file=sys.stderr)
        print(f"  {category:52s} +{len(found) - before}")

    with CANDIDATES.open("w") as fh:
        for rec in found.values():
            fh.write(json.dumps(rec) + "\n")
    return list(found.values())


def download_one(record: dict, tries: int = 4) -> Optional[dict]:
    """Fetch one thumbnail, backing off when Commons pushes back."""
    dest = POOL_DIR / f"{record['sha1']}.jpg"
    record["path"] = str(dest.relative_to(ROOT))
    if dest.exists() and dest.stat().st_size > 4096:
        record["cached"] = True
        return record

    for attempt in range(tries):
        try:
            req = urllib.request.Request(record["thumburl"], headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            if len(blob) < 4096:
                return None
            dest.write_bytes(blob)
            record["cached"] = False
            return record
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                time.sleep(DOWNLOAD_DELAY * (2 ** (attempt + 1)) + random.random())
                continue
            return None
        except Exception:  # noqa: BLE001 - one bad file must not kill the run
            time.sleep(DOWNLOAD_DELAY)
    return None


def cmd_collect(args: argparse.Namespace) -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    if CANDIDATES.exists() and not args.refresh:
        candidates = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
        print(f"using cached listing: {len(candidates)} candidates "
              f"(--refresh to re-query Commons)")
    else:
        print("listing categories...")
        candidates = build_candidates()
        print(f"listed {len(candidates)} candidates")

    have: set = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                have.add(json.loads(line)["sha1"])
    print(f"manifest already holds {len(have)} images")

    # Per-pool caps: keep every wet/mixed frame, subsample the abundant dry pool.
    rng = random.Random(args.seed)
    by_hint: Dict[str, List[dict]] = {}
    for rec in candidates:
        if rec["sha1"] in have:
            continue
        by_hint.setdefault(rec.get("hint", "dry"), []).append(rec)

    pending: List[dict] = []
    for hint, items in by_hint.items():
        rng.shuffle(items)
        cap = HINT_CAPS.get(hint)
        chosen = items if cap is None else items[:cap]
        pending.extend(chosen)
        print(f"  pool {hint:6s}: {len(chosen)} of {len(items)} available")

    # Interleave the pools. Downloading them in plan order means an interrupted
    # run yields every wet image and no dry ones - unusable. Shuffled, a partial
    # pool is still a proportional sample of all four classes.
    rng.shuffle(pending)

    if args.limit:
        pending = pending[: args.limit]

    eta = len(pending) * (DOWNLOAD_DELAY + 0.7) / 60.0
    print(f"\ndownloading {len(pending)} images serially (~{eta:.0f} min)...")

    saved = failed = 0
    with MANIFEST.open("a") as fh:
        for i, rec in enumerate(pending, 1):
            got = download_one(rec)
            if got:
                fh.write(json.dumps(got) + "\n")
                fh.flush()  # durable progress: a kill mid-run loses nothing
                saved += 1
            else:
                failed += 1
            time.sleep(DOWNLOAD_DELAY)
            if i % 50 == 0:
                print(f"    {i}/{len(pending)}  saved={saved} failed={failed}", flush=True)

    print(f"\nsaved {saved}, failed {failed}")
    print(f"pool now: {len(list(POOL_DIR.glob('*.jpg')))} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="find candidate Commons categories")
    d.add_argument("--terms", nargs="*")
    d.add_argument("--min-files", type=int, default=40)
    d.set_defaults(func=cmd_discover)

    c = sub.add_parser("collect", help="download the image pool")
    c.add_argument("--refresh", action="store_true", help="re-query the category listing")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--seed", type=int, default=1337)
    c.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
