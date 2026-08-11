"""Assemble the reviewed pool into an imagefolder-compatible dataset.

Final label = human correction where one exists, CLIP triage otherwise.
Rejects are dropped, classes are capped so one bucket cannot dominate.

Two filters keep known-bad labels out of the training set:

`--min-confidence` drops unreviewed images whose CLIP call was weak. Review
showed the low-confidence band is where Dry, Damp and Wet blur together, so an
unchecked label there is close to a coin flip. Human-reviewed images bypass this
filter entirely, whatever CLIP thought.

`--min-verified` drops any class that too few humans confirmed. This is what
removes `Drying`: CLIP assigned 61 images to it and review found none of them
were drying tracks (they were aerial circuit maps and sunny Monaco streets), so
the class has no verified support and must not be trained on. Drying is still
produced by the system, from the trend layer, which is where it comes from.

Split policy: reviewed images go to test and validation first. Evaluation
numbers are only worth quoting if the labels behind them were checked by hand,
so the scarce verified labels are spent on eval rather than training.

Outputs:
    data/dataset/{train,validation,test}/<Class>/*.jpg
    data/dataset/ATTRIBUTION.csv     per-image credit (required by CC-BY)
    data/dataset/stats.json          counts, licences, CLIP disagreement rate

Usage:
    python -m data_pipeline.build_dataset --max-per-class 260
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from app.labels import CLASSES, canonical

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
PRESORT = ROOT / "data" / "raw" / "presort.jsonl"
REVIEW_DIR = ROOT / "data" / "review"
OUT = ROOT / "data" / "dataset"

REJECT = "Reject"
SPLITS = ("train", "validation", "test")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        help="drop unreviewed images below this CLIP confidence")
    parser.add_argument("--min-verified", type=int, default=5,
                        help="drop classes with fewer than this many human-confirmed images")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    presort = load_jsonl(PRESORT)
    manifest = {r["sha1"]: r for r in load_jsonl(MANIFEST)}

    corrections_path = REVIEW_DIR / "corrections.json"
    corrections: Dict[str, str] = {}
    if corrections_path.exists():
        corrections = json.loads(corrections_path.read_text() or "{}")

    # "Reviewed" means a human rendered a verdict on this image, which is the
    # key set of corrections.json, confirmations included. Counting every cell
    # that merely appeared on a contact sheet would overstate verification and
    # let unchecked labels into the test split.
    reviewed = set(corrections)

    # --- resolve final labels ---------------------------------------------
    by_class: Dict[str, List[dict]] = defaultdict(list)
    disagreements = 0
    skipped_low_conf = 0

    for rec in presort:
        sha = rec["sha1"]
        is_reviewed = sha in reviewed
        raw = corrections.get(sha, rec["clip_label"])

        if is_reviewed and raw != rec["clip_label"]:
            disagreements += 1
        if raw == REJECT:
            continue
        if not is_reviewed and rec["clip_confidence"] < args.min_confidence:
            skipped_low_conf += 1
            continue

        label = canonical(raw)
        by_class[label].append(dict(rec, label=label, reviewed=is_reviewed))

    # --- drop classes without enough human-confirmed support ---------------
    verified_counts = {
        cls: sum(1 for r in items if r["reviewed"]) for cls, items in by_class.items()
    }
    kept = [c for c in CLASSES
            if c in by_class and verified_counts.get(c, 0) >= args.min_verified]
    dropped = [c for c in CLASSES if c in by_class and c not in kept]
    for cls in dropped:
        print(f"dropping class {cls!r}: only {verified_counts.get(cls, 0)} "
              f"human-verified images (need {args.min_verified})")
        by_class.pop(cls, None)

    if not kept:
        raise SystemExit("no classes survived filtering - review more images first")

    # --- cap + split -------------------------------------------------------
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        for cls in kept:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in kept} for s in SPLITS}
    attribution_rows = []

    for cls in kept:
        items = by_class[cls]
        rng.shuffle(items)
        # Reviewed first so they land in test/validation.
        items.sort(key=lambda r: not r["reviewed"])
        items = items[: args.max_per_class]

        n = len(items)
        n_test = max(1, int(n * args.test_frac)) if n else 0
        n_val = max(1, int(n * args.val_frac)) if n else 0
        assignment = (
            ["test"] * n_test + ["validation"] * n_val + ["train"] * (n - n_test - n_val)
        )

        for rec, split in zip(items, assignment):
            src = ROOT / rec["path"]
            if not src.exists():
                continue
            dst = OUT / split / cls / f"{rec['sha1'][:16]}.jpg"
            shutil.copyfile(src, dst)
            counts[split][cls] += 1

            meta = manifest.get(rec["sha1"], {})
            attribution_rows.append(
                {
                    "file": str(dst.relative_to(OUT)),
                    "label": cls,
                    "split": split,
                    "title": meta.get("title", ""),
                    # A blank credit would quietly break the CC-BY condition.
                    "artist": meta.get("artist") or "Wikimedia Commons contributor",
                    "license": meta.get("license", ""),
                    "license_url": meta.get("license_url", ""),
                    "source_url": meta.get("descriptionurl", ""),
                }
            )

    # --- attribution + stats ----------------------------------------------
    with (OUT / "ATTRIBUTION.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file", "label", "split", "title", "artist",
                        "license", "license_url", "source_url"],
        )
        writer.writeheader()
        writer.writerows(attribution_rows)

    total = sum(counts[s][c] for s in SPLITS for c in kept)
    rate = (disagreements / len(reviewed)) if reviewed else 0.0
    stats = {
        "total_images": total,
        "classes": kept,
        "dropped_classes": dropped,
        "counts": counts,
        "reviewed_images": len(reviewed),
        "clip_wrong_on_reviewed": disagreements,
        "clip_disagreement_rate": round(rate, 4),
        "skipped_low_confidence": skipped_low_conf,
        "min_confidence": args.min_confidence,
        "license_breakdown": {},
    }
    for row in attribution_rows:
        lic = row["license"] or "unknown"
        stats["license_breakdown"][lic] = stats["license_breakdown"].get(lic, 0) + 1

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\ndataset -> {OUT}  ({total} images across {len(kept)} classes)")
    print("split".ljust(12) + "".join(c.ljust(9) for c in kept))
    for split in SPLITS:
        print(split.ljust(12) + "".join(str(counts[split][c]).ljust(9) for c in kept))
    print(f"\nhuman-reviewed: {len(reviewed)}   CLIP wrong on {disagreements} "
          f"({rate:.1%})   dropped {skipped_low_conf} low-confidence unreviewed")
    print(f"licences: {stats['license_breakdown']}")


if __name__ == "__main__":
    main()
