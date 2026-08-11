"""Assemble the bundled demo sequence: a wet -> drying -> dry progression.

Honest about what this is: the frames are **real** Commons trackside photographs,
each independently labelled, ordered into a plausible weather progression. It is
not a continuous video clip - Commons has no wet-to-dry video of a single
session under a redistributable licence.

That is enough to demonstrate the thing being demonstrated. The trend layer
consumes a *sequence of frame predictions*; it neither knows nor cares whether
consecutive frames came from one camera. Judges can equally drop in their own
video frames - the path is identical.

Usage:
    python -m data_pipeline.make_demo_sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

from app.labels import CLASS_DAMP, CLASS_DRY, CLASS_DRYING, CLASS_WET, canonical

ROOT = Path(__file__).resolve().parents[1]
PRESORT = ROOT / "data" / "raw" / "presort.jsonl"
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
REVIEW = ROOT / "data" / "review" / "corrections.json"
DEMO_DIR = ROOT / "data" / "demo"

#: Soaked, then drying back, then usable. `Drying` is absent on purpose: it is
#: not a frame class (see docs/JUDGES.md), it is what the trend layer should
#: *report* while this sequence plays through Wet -> Damp -> Dry.
STORY: List[str] = [CLASS_WET] * 5 + [CLASS_DAMP] * 4 + [CLASS_DRY] * 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not PRESORT.exists():
        raise SystemExit("run data_pipeline.presort_clip first")

    corrections: Dict[str, str] = {}
    if REVIEW.exists():
        corrections = json.loads(REVIEW.read_text() or "{}")
    manifest = {
        json.loads(l)["sha1"]: json.loads(l)
        for l in MANIFEST.read_text().splitlines() if l.strip()
    }

    # Prefer human-confirmed labels; they make the demo predictable.
    pools: Dict[str, List[dict]] = {}
    for line in PRESORT.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        raw = corrections.get(rec["sha1"], rec["clip_label"])
        if raw == "Reject":
            continue
        rec["reviewed"] = rec["sha1"] in corrections
        rec["final"] = canonical(raw)
        pools.setdefault(rec["final"], []).append(rec)

    rng = random.Random(args.seed)
    for label, items in pools.items():
        # Reviewed first, then most confident.
        items.sort(key=lambda r: (not r["reviewed"], -r["clip_confidence"]))

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    used: set = set()
    rows = []
    missing: List[str] = []
    index = 0
    for label in STORY:
        candidates = [r for r in pools.get(label, []) if r["sha1"] not in used]
        if not candidates:
            missing.append(label)
            continue
        pick = candidates[0]
        used.add(pick["sha1"])
        index += 1
        dest = DEMO_DIR / f"{index:02d}_{label.lower()}.jpg"
        shutil.copyfile(ROOT / pick["path"], dest)
        meta = manifest.get(pick["sha1"], {})
        rows.append(
            {
                "file": dest.name,
                "label": label,
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "license": meta.get("license", ""),
                "source_url": meta.get("descriptionurl", ""),
            }
        )

    with (DEMO_DIR / "ATTRIBUTION.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["file", "label", "title", "artist", "license", "source_url"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"demo sequence -> {DEMO_DIR} ({len(rows)} frames)")
    for row in rows:
        print(f"  {row['file']:24s} {row['label']}")
    if missing:
        print(f"\n[!] no images available for: {sorted(set(missing))}")


if __name__ == "__main__":
    main()
