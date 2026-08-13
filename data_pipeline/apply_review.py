"""Merge contact-sheet verdicts into corrections.json.

Reviewing produces judgements keyed by the cell ids printed on the sheets
(`BN003`, `DMP012`, ...). This maps them back to image hashes and merges them
into `data/review/corrections.json`, which is the durable label store.

Write the verdicts as JSON, either one entry per cell:

    {"BN000": "Dry", "BN002": "Damp", "BN006": "Wet"}

or a whole sheet in reading order, left to right, top to bottom:

    {"BN0": ["Dry", "Reject", "Damp", ...]}

Confirmations are recorded as well as changes, because "a human looked at this
and agreed" is what makes an image eligible for the test split.

Usage:
    python -m data_pipeline.apply_review verdicts.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

from app.labels import CLASSES

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "data" / "review"
PRESORT = ROOT / "data" / "raw" / "presort.jsonl"

REJECT = "Reject"
VALID = set(CLASSES) | {REJECT}


def expand(raw: Dict[str, object]) -> Dict[str, str]:
    """Turn prefix->list shorthand into explicit cell-id->label pairs."""
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            for i, label in enumerate(value):
                out[f"{key}{i:02d}"] = label
        else:
            out[key] = str(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts", help="JSON file of cell-id -> label")
    args = parser.parse_args()

    verdicts = expand(json.loads(Path(args.verdicts).read_text()))

    bad = {c: l for c, l in verdicts.items() if l not in VALID}
    if bad:
        raise SystemExit(f"unknown labels: {bad}\nvalid: {sorted(VALID)}")

    index_map = json.loads((REVIEW_DIR / "index_map.json").read_text())
    corrections_path = REVIEW_DIR / "corrections.json"
    corrections = json.loads(corrections_path.read_text() or "{}")

    applied, missing = 0, []
    for cell, label in verdicts.items():
        sha = index_map.get(cell)
        if not sha:
            missing.append(cell)
            continue
        corrections[sha] = label
        applied += 1

    corrections_path.write_text(json.dumps(corrections, indent=1))

    presort = {
        json.loads(l)["sha1"]: json.loads(l)["clip_label"]
        for l in PRESORT.read_text().splitlines() if l.strip()
    }
    wrong = sum(1 for s, v in corrections.items() if presort.get(s) != v)

    print(f"applied {applied} verdicts ({len(corrections)} decisions total)")
    if missing:
        print(f"[!] {len(missing)} cell ids not on the current sheets: {missing[:8]}")
    print(f"CLIP disagreed with review on {wrong}/{len(corrections)} "
          f"({wrong / len(corrections):.1%})")
    print("label counts:", dict(Counter(corrections.values())))


if __name__ == "__main__":
    main()
