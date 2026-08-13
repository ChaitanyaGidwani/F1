"""Render numbered contact sheets so the CLIP triage can be checked by eye.

Reviewing thousands of images one at a time is not practical; reviewing 25 at a
time on a labelled grid is. Each cell carries a short id, and the sheet ships
with a sidecar JSON mapping cell id -> sha1, so corrections are written back as

    data/review/corrections.json   {"<sha1>": "Wet" | "Dry" | ... | "Reject"}

Anything without an entry keeps its CLIP label; entries override it.

Two selection modes:

`buckets` (default) walks CLIP's own class buckets, most confident first. Good
for a first pass, since the confident images define each class.

`boundary` samples the *least* confident Dry and Wet calls from wet-weekend
imagery. This is where Damp actually lives: measured yield is around 35% Damp
per sheet, against 23% in the bucket CLIP itself labels "Damp". Already-decided
images are skipped, so repeated runs keep reaching new candidates.

Usage:
    python -m data_pipeline.make_contactsheets
    python -m data_pipeline.make_contactsheets --mode boundary --sheets 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

from app.labels import CLASSES

ROOT = Path(__file__).resolve().parents[1]
PRESORT = ROOT / "data" / "raw" / "presort.jsonl"
REVIEW_DIR = ROOT / "data" / "review"

REJECT = "Reject"

#: Explicit per-class cell-id prefixes. Deriving these from the class name
#: (e.g. `cls[:2]`) silently collides "Dry" with "Drying" - both give "DR" -
#: so two different images share a cell id and one overwrites the other in
#: index_map.json, corrupting labels invisibly. Keep these distinct.
PREFIXES = {"Dry": "DRY", "Damp": "DMP", "Drying": "DRG", "Wet": "WET", REJECT: "REJ"}

CELL = 224
CAPTION = 22
COLS = 5

Sheet = Tuple[str, List[dict], str]  # (name, records, cell-id prefix)


def build_sheet(records: List[dict], out_path: Path, index_map: Dict[str, str]) -> None:
    rows = (len(records) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + CAPTION)), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)

    for i, rec in enumerate(records):
        x, y = (i % COLS) * CELL, (i // COLS) * (CELL + CAPTION)
        try:
            img = Image.open(ROOT / rec["path"]).convert("RGB")
            sheet.paste(img.resize((CELL, CELL), Image.BILINEAR), (x, y))
        except Exception:  # noqa: BLE001
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 20, 20))

        cell_id = rec["cell_id"]
        index_map[cell_id] = rec["sha1"]
        caption = f"{cell_id}  {rec['clip_label'][:7]} {rec['clip_confidence']:.2f}"
        draw.rectangle([x, y + CELL, x + CELL, y + CELL + CAPTION], fill=(10, 10, 12))
        draw.text((x + 4, y + CELL + 5), caption, fill=(235, 235, 240))
        # id badge on the image itself, so a cell stays identifiable when cropped
        draw.rectangle([x + 2, y + 2, x + 52, y + 20], fill=(0, 0, 0))
        draw.text((x + 6, y + 6), cell_id, fill=(255, 220, 90))

    sheet.save(out_path, quality=88)


def select_buckets(records: List[dict], per_sheet: int, max_sheets: int) -> List[Sheet]:
    by_class: Dict[str, List[dict]] = {c: [] for c in CLASSES + [REJECT]}
    for rec in records:
        by_class.setdefault(rec["clip_label"], []).append(rec)

    sheets: List[Sheet] = []
    for cls, items in by_class.items():
        # Most confident first: those define the class, so errors there hurt most.
        items.sort(key=lambda r: -r["clip_confidence"])
        prefix = PREFIXES.get(cls, cls[:3].upper())
        n = min(max_sheets, (len(items) + per_sheet - 1) // per_sheet)
        for i in range(n):
            chunk = items[i * per_sheet : (i + 1) * per_sheet]
            if chunk:
                sheets.append((f"{cls}_{i}", chunk, f"{prefix}{i}"))
    return sheets


def select_boundary(
    records: List[dict], decided: set, per_sheet: int, sheets_wanted: int
) -> List[Sheet]:
    cand = [
        r for r in records
        if r["sha1"] not in decided
        and r.get("hint") == "mixed"
        and r["clip_label"] in ("Wet", "Dry")
    ]
    cand.sort(key=lambda r: r["clip_confidence"])
    sheets: List[Sheet] = []
    for i in range(sheets_wanted):
        chunk = cand[i * per_sheet : (i + 1) * per_sheet]
        if not chunk:
            break
        sheets.append((f"boundary_{i}", chunk, f"BN{i}"))
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["buckets", "boundary"], default="buckets")
    parser.add_argument("--per-sheet", type=int, default=25)
    parser.add_argument("--max-sheets-per-class", type=int, default=8)
    parser.add_argument("--sheets", type=int, default=4, help="boundary mode only")
    args = parser.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in PRESORT.read_text().splitlines() if l.strip()]

    corrections_path = REVIEW_DIR / "corrections.json"
    decided = set(json.loads(corrections_path.read_text() or "{}")) if corrections_path.exists() else set()

    if args.mode == "boundary":
        plan = select_boundary(records, decided, args.per_sheet, args.sheets)
    else:
        plan = select_buckets(records, args.per_sheet, args.max_sheets_per_class)

    index_map: Dict[str, str] = {}
    manifest = []
    for name, chunk, prefix in plan:
        for j, rec in enumerate(chunk):
            rec["cell_id"] = f"{prefix}{j:02d}"
        out = REVIEW_DIR / f"sheet_{name}.jpg"
        build_sheet(chunk, out, index_map)
        manifest.append({"sheet": out.name, "count": len(chunk), "prefix": prefix})
        print(f"  {out.name}  ({len(chunk)} cells, ids {prefix}00-{prefix}{len(chunk)-1:02d})")

    (REVIEW_DIR / "index_map.json").write_text(json.dumps(index_map, indent=2))
    (REVIEW_DIR / "sheets.json").write_text(json.dumps(manifest, indent=2))
    if not corrections_path.exists():
        corrections_path.write_text("{}\n")

    print(f"\n{len(plan)} sheets -> {REVIEW_DIR}")
    print(f"{len(index_map)} cells; already-decided images were skipped"
          if args.mode == "boundary" else f"{len(index_map)} cells")


if __name__ == "__main__":
    main()
