"""Push the fine-tuned classifier to the Hub with a model card.

    huggingface-cli login
    python -m training.push_model --repo-id <your-org>/vit-track-condition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "vit-track-condition"

CARD = """---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- image-classification
- motorsport
- racing
- track-conditions
- weather
pipeline_tag: image-classification
---

# ViT - Trackside Track-Condition Classifier

Classifies a racing-surface frame as **Dry / Damp / Drying / Wet**.

Fine-tuned from `google/vit-base-patch16-224-in21k` on trackside motorsport
imagery collected from Wikimedia Commons. Built for *Weather Whiplash*, a live
track-condition detector for race strategy.

## Usage

```python
from transformers import pipeline

pipe = pipeline("image-classification", model="{repo_id}")
pipe("trackside_frame.jpg")
```

## Why fine-tune instead of zero-shot

CLIP zero-shot with surface-worded prompts is a reasonable baseline and is kept
in the project as a fallback, but it confuses an overcast sky with a wet surface
and cannot represent `Drying` (a dry racing line on wet asphalt) at all -
there is no natural-language prompt that reliably isolates it.

{eval_section}

## Intended use and limits

Decision *support* for race strategy, not an autonomous decision-maker. A single
frame is never enough to call "drying" - that requires the trend layer in the
project repo, which reads direction of change across a rolling window of frames.

Training data skews to Formula One and European circuits. Expect degradation on
night races, heavy motion blur, unusual surfaces (street circuits, gravel) and
other series.

## Training data

See the companion dataset. Images are CC-BY / CC-BY-SA from Wikimedia Commons
with per-image attribution retained.
"""


def eval_section() -> str:
    path = MODEL_DIR / "eval_results.json"
    if not path.exists():
        return "## Evaluation\n\nNot yet evaluated."
    results = json.loads(path.read_text())
    lines = ["## Evaluation", ""]
    for split in ("validation", "test"):
        metrics = results.get(split)
        if not metrics:
            continue
        acc = metrics.get(f"{split}_accuracy")
        f1 = metrics.get(f"{split}_f1_macro")
        if acc is not None:
            lines.append(f"- **{split}**: accuracy {acc:.3f}, macro-F1 {f1:.3f}")

    report = results.get("test_report")
    if isinstance(report, dict):
        lines += ["", "Per-class on the held-out test split:", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for name, row in report.items():
            if isinstance(row, dict) and "f1-score" in row and name not in (
                "accuracy", "macro avg", "weighted avg"
            ):
                lines.append(
                    f"| {name} | {row['precision']:.2f} | {row['recall']:.2f} | "
                    f"{row['f1-score']:.2f} | {int(row['support'])} |"
                )
    lines += ["", "The test split is hand-verified, so these numbers are measured "
              "against human labels rather than triage output."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit("models/vit-track-condition missing - train first")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    card = CARD.format(repo_id=args.repo_id, eval_section=eval_section())
    (MODEL_DIR / "README.md").write_text(card)

    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "runs/*", "*.pt"],
    )
    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
