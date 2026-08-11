"""Push the assembled dataset to the Hugging Face Hub, with a real card.

Run this yourself so the token stays yours:

    source .venv/bin/activate && hf auth login
    python -m data_pipeline.push_dataset --repo-id <your-org>/trackside-condition

See docs/PUSH_TO_HUB.md for the org setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from app.labels import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset"


CARD = """---
license: cc-by-sa-4.0
task_categories:
- image-classification
tags:
- motorsport
- racing
- weather
- road-surface
- track-conditions
size_categories:
- n<1K
---

# Trackside Track-Condition Dataset

Frame-level racing-surface condition labels for motorsport imagery:
**Dry / Damp / Wet**.

Built for the *Weather Whiplash* project - a live track-condition detector that
tells a pit wall whether the circuit is getting safer or riskier, and therefore
when to change tyres.

## Why this dataset exists

The nearest existing resource is **RSCD** (Road Surface Classification Dataset),
an autonomous-driving dataset of dry / wet / water road patches. It is a useful
signal but sits in a different visual domain:

| | RSCD | This dataset |
|---|---|---|
| Camera | bumper-height, car-mounted | trackside / broadcast / onboard |
| Framing | close-up road patch, surface fills the frame | wide, surface is part of a scene |
| Surface | public asphalt | racing asphalt, painted kerbs, run-off, gravel traps |
| Confounders | traffic, lane markings | spray plumes, tyre marbles, a drying racing line |

A classifier trained only on car-perspective road patches has to generalise
across all four rows at once. This dataset exists to close that gap.

## Classes

| Class | Definition |
|---|---|
| `Dry` | Uniform pale-grey asphalt, no sheen |
| `Damp` | Uniformly darkened asphalt, no standing water, no spray |
| `Wet` | Standing water, strong reflections, visible spray |

There is no `Drying` class, and its absence is a finding rather than an
oversight. Of the 61 images CLIP assigned to "drying", none of the 25 checked
were drying tracks: they were aerial circuit maps and sunny street scenes. With
no verified examples the class was dropped automatically.

Drying is a property of a sequence, not a frame. A damp track and a drying track
can look identical in one image; the difference is the direction of travel over
the previous few frames. The project derives it from a trend layer over
consecutive predictions instead.

## Sourcing and licensing

Every image comes from **Wikimedia Commons**, primarily the
`Formula One in rain in the <decade>s` category tree plus wet race weekends and
circuit categories. Broadcast screenshots were avoided because they cannot
be redistributed, which would make this dataset unpublishable.

Per-image title, author, licence and source URL are in **`ATTRIBUTION.csv`**.
Licences are CC-BY / CC-BY-SA / public domain; the aggregate is released as
CC-BY-SA-4.0. **If you use this dataset, retain the attribution file.**

## Labelling method

1. Candidate pool collected from Commons categories (weak category-level prior).
2. CLIP zero-shot triage into the four classes plus a reject bucket
   (paddock portraits, garages, crowds - anything where the surface is not
   visible).
3. **Human review of contact sheets**, correcting the triage. Corrections
   override CLIP.

Step 3 is what stops this being a distillation of CLIP. Hand-verified images are
allocated to the test and validation splits first, so reported accuracy is
measured against checked labels.

{stats_section}

## Limitations

- Skewed toward Formula One and European circuits; other series and surfaces
  are thinly represented.
- `Damp` is the hardest class for annotators and for models. Its boundaries with
  both `Dry` and `Wet` are genuinely continuous, and it is by far the least
  represented, because photographers shoot dramatic conditions and a merely damp
  track is not dramatic.
- Photographs are composed shots, not uniformly sampled video frames, so the
  distribution is not identical to a real fixed trackside camera feed.
"""


def stats_section() -> str:
    stats_path = DATA_DIR / "stats.json"
    if not stats_path.exists():
        return "## Splits\n\nRun `python -m data_pipeline.build_dataset` first."
    stats = json.loads(stats_path.read_text())
    counts = stats["counts"]
    # Use the classes the build actually kept, not the full vocabulary; a
    # dropped class would otherwise show as a column of zeroes.
    classes = stats.get("classes") or CLASSES
    lines = ["## Splits", "", "| Split | " + " | ".join(classes) + " | Total |",
             "|---|" + "---|" * (len(classes) + 1)]
    for split, per_class in counts.items():
        total = sum(per_class.values())
        lines.append(
            f"| {split} | " + " | ".join(str(per_class.get(c, 0)) for c in classes)
            + f" | {total} |"
        )
    lines += [
        "",
        f"Total images: **{stats['total_images']}**  ",
        f"Human-reviewed: **{stats['reviewed_images']}**  ",
        f"CLIP triage corrected by a human on **{stats['clip_disagreement_rate']:.1%}** "
        "of reviewed images.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. my-org/trackside-condition")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit("data/dataset missing - run data_pipeline.build_dataset first")

    ds = load_dataset("imagefolder", data_dir=str(DATA_DIR))
    print({k: len(v) for k, v in ds.items()})

    print(f"pushing to {args.repo_id} ...")
    ds.push_to_hub(args.repo_id, private=args.private)

    api = HfApi()
    card = CARD.format(stats_section=stats_section())
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    attribution = DATA_DIR / "ATTRIBUTION.csv"
    if attribution.exists():
        api.upload_file(
            path_or_fileobj=str(attribution),
            path_in_repo="ATTRIBUTION.csv",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"done -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
