"""Render numbered contact sheets so the CLIP triage can be checked by eye.

Reviewing ~2000 images one at a time is not practical; reviewing 25 at a time on
a labelled grid is. Each cell carries a short id, and the sheet ships with a
sidecar JSON mapping cell id -> sha1, so corrections can be written back as

    data/review/corrections.json   {"<sha1>": "Wet" | "Dry" | ... | "Reject"}

Anything without a correction keeps its CLIP label; corrections override it.

Usage:
    python -m data_pipeline.make_contactsheets --per-sheet 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def build_sheet(records: List[dict], out_path: Path, index_map: Dict[str, str]) -> None:
    rows = (len(records) + COLS - 1) // COLS
    width = COLS * CELL
    height = rows * (CELL + CAPTION)
    sheet = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)

    for i, rec in enumerate(records):
        col, row = i % COLS, i // COLS
        x, y = col * CELL, row * (CELL + CAPTION)
        try:
            img = Image.open(ROOT / rec["path"]).convert("RGB")
            img = img.resize((CELL, CELL), Image.BILINEAR)
            sheet.paste(img, (x, y))
        except Exception:  # noqa: BLE001
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 20, 20))

        cell_id = rec["cell_id"]
        index_map[cell_id] = rec["sha1"]
        caption = f"{cell_id}  {rec['clip_label'][:7]} {rec['clip_confidence']:.2f}"
        draw.rectangle([x, y + CELL, x + CELL, y + CELL + CAPTION], fill=(10, 10, 12))
        draw.text((x + 4, y + CELL + 5), caption, fill=(235, 235, 240))
        # id badge on the image itself, so a cell is identifiable when cropped
        draw.rectangle([x + 2, y + 2, x + 46, y + 20], fill=(0, 0, 0))
        draw.text((x + 6, y + 6), cell_id, fill=(255, 220, 90))

    sheet.save(out_path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sheet", type=int, default=25)
    parser.add_argument("--max-sheets-per-class", type=int, default=8)
    args = parser.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in PRESORT.read_text().splitlines() if l.strip()]

    by_class: Dict[str, List[dict]] = {c: [] for c in CLASSES + [REJECT]}
    for rec in records:
        by_class.setdefault(rec["clip_label"], []).append(rec)

    # Review the most confident first: those set the class definition, and errors
    # among them are the most damaging.
    index_map: Dict[str, str] = {}
    manifest = []
    for cls, items in by_class.items():
        items.sort(key=lambda r: -r["clip_confidence"])
        prefix = PREFIXES.get(cls, cls[:3].upper())
        for sheet_no in range(min(args.max_sheets_per_class,
                                  (len(items) + args.per_sheet - 1) // args.per_sheet)):
            chunk = items[sheet_no * args.per_sheet : (sheet_no + 1) * args.per_sheet]
            if not chunk:
                break
            for j, rec in enumerate(chunk):
                rec["cell_id"] = f"{prefix}{sheet_no}{j:02d}"
            out = REVIEW_DIR / f"sheet_{cls}_{sheet_no}.jpg"
            build_sheet(chunk, out, index_map)
            manifest.append({"sheet": out.name, "clip_label": cls, "count": len(chunk)})
            print(f"  {out.name}  ({len(chunk)} cells)")

    (REVIEW_DIR / "index_map.json").write_text(json.dumps(index_map, indent=2))
    (REVIEW_DIR / "sheets.json").write_text(json.dumps(manifest, indent=2))
    corrections = REVIEW_DIR / "corrections.json"
    if not corrections.exists():
        corrections.write_text("{}\n")
    print(f"\n{len(manifest)} sheets -> {REVIEW_DIR}")
    print(f"cell ids -> {REVIEW_DIR / 'index_map.json'}")


if __name__ == "__main__":
    main()
