"""Compare the fine-tuned classifier against the CLIP zero-shot baseline.

This produces the number the whole project rests on. "We fine-tuned a model" is
only interesting if fine-tuning bought something measurable over the zero-shot
tool anyone could have called in an afternoon - so both are evaluated on the
same hand-verified test split and reported side by side.

    python -m training.evaluate

Writes models/vit-track-condition/comparison.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.clip_prompts import CLASS_PROMPTS
from app.labels import canonical

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset"
MODEL_DIR = ROOT / "models" / "vit-track-condition"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% confidence interval for an accuracy, Wilson score method.

    Quoting a bare accuracy on a test set this small invites a reader to believe
    it to three decimal places. The interval is the honest number.
    """
    if not n:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar(y_true: List[int], a_pred: List[int], b_pred: List[int]):
    """Exact McNemar test on two models scored over the *same* items.

    Comparing two independent accuracy figures ignores that both models saw the
    same images, which is the information that makes a small test set usable.
    Only the disagreements carry signal: cases both got right or both got wrong
    tell you nothing about which is better.
    """
    only_a = sum(1 for t, a, b in zip(y_true, a_pred, b_pred) if a == t and b != t)
    only_b = sum(1 for t, a, b in zip(y_true, a_pred, b_pred) if a != t and b == t)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    p = min(1.0, 2 * tail / (2**n))
    return only_a, only_b, p


def print_confusion(y_true: List[int], y_pred: List[int], names: List[str], title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    width = max(len(n) for n in names) + 2
    print(f"\n{title} (rows = truth, cols = predicted)")
    print(" " * width + "".join(n[:6].rjust(8) for n in names))
    for name, row in zip(names, matrix):
        print(name.ljust(width) + "".join(str(v).rjust(8) for v in row))


def evaluate_finetuned(images, y_true, names) -> Dict[str, object]:
    from transformers import pipeline

    pipe = pipeline("image-classification", model=str(MODEL_DIR), device=device())
    y_pred = []
    for image in images:
        scores = pipe(image, top_k=None)
        best = max(scores, key=lambda s: s["score"])
        y_pred.append(names.index(canonical(best["label"])))
    return {"y_pred": y_pred}


def evaluate_clip(images, y_true, names, model_id: str) -> Dict[str, object]:
    from transformers import CLIPModel, CLIPProcessor

    dev = device()
    model = CLIPModel.from_pretrained(model_id).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    # Restrict CLIP to the label space the dataset actually uses, otherwise it
    # can predict a class the fine-tuned model was never allowed to output and
    # the comparison stops being like-for-like.
    prompts, owner = [], []
    for cls in names:
        for prompt in CLASS_PROMPTS[cls]:
            prompts.append(prompt)
            owner.append(cls)

    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(dev)
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    y_pred = []
    for start in range(0, len(images), 32):
        batch = images[start : start + 32]
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(dev)
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sims = emb @ text_emb.T
        for row in sims:
            best_cls = max(
                names,
                key=lambda c: max(
                    float(row[i]) for i, o in enumerate(owner) if o == c
                ),
            )
            y_pred.append(names.index(best_cls))
    return {"y_pred": y_pred}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    ds = load_dataset("imagefolder", data_dir=str(DATA_DIR))[args.split]
    names = ds.features["label"].names
    images = [row["image"].convert("RGB") for row in ds]
    y_true = list(ds["label"])
    print(f"evaluating on {len(images)} hand-verified {args.split} images")
    print(f"classes: {names}")

    results: Dict[str, object] = {"split": args.split, "n": len(images)}

    print("\n" + "=" * 62)
    print("CLIP zero-shot baseline")
    print("=" * 62)
    clip_pred = evaluate_clip(images, y_true, names, args.clip_model)["y_pred"]
    clip_acc = accuracy_score(y_true, clip_pred)
    print(classification_report(y_true, clip_pred, target_names=names, zero_division=0))
    print_confusion(y_true, clip_pred, names, "CLIP zero-shot")
    results["clip"] = {
        "accuracy": clip_acc,
        "report": classification_report(
            y_true, clip_pred, target_names=names, zero_division=0, output_dict=True
        ),
    }

    if not MODEL_DIR.exists():
        print("\n[!] no fine-tuned model found - train first for the comparison")
        results["finetuned"] = None
    else:
        print("\n" + "=" * 62)
        print("Fine-tuned ViT")
        print("=" * 62)
        ft_pred = evaluate_finetuned(images, y_true, names)["y_pred"]
        ft_acc = accuracy_score(y_true, ft_pred)
        print(classification_report(y_true, ft_pred, target_names=names, zero_division=0))
        print_confusion(y_true, ft_pred, names, "Fine-tuned ViT")
        results["finetuned"] = {
            "accuracy": ft_acc,
            "report": classification_report(
                y_true, ft_pred, target_names=names, zero_division=0, output_dict=True
            ),
        }

        n = len(y_true)
        clip_lo, clip_hi = wilson_interval(round(clip_acc * n), n)
        ft_lo, ft_hi = wilson_interval(round(ft_acc * n), n)
        only_ft, only_clip, p_value = mcnemar(y_true, ft_pred, clip_pred)

        print("\n" + "=" * 62)
        print(f"  CLIP zero-shot : {clip_acc:.1%}  95% CI [{clip_lo:.1%}, {clip_hi:.1%}]")
        print(f"  Fine-tuned ViT : {ft_acc:.1%}  95% CI [{ft_lo:.1%}, {ft_hi:.1%}]")
        print(f"  Difference     : {ft_acc - clip_acc:+.1%} on n={n}")
        print("-" * 62)
        print("  McNemar paired test (same images, disagreements only)")
        print(f"    ViT right, CLIP wrong : {only_ft}")
        print(f"    CLIP right, ViT wrong : {only_clip}")
        print(f"    exact two-sided p     : {p_value:.3f}")
        verdict = ("significant at 0.05" if p_value < 0.05
                   else "NOT significant - do not claim an overall accuracy win")
        print(f"    verdict               : {verdict}")
        print("-" * 62)

        # Per-class recall is where the two models actually differ, and it does
        # not depend on the overall accuracy gap being significant. Each class
        # gets its own paired test, because "the model finds damp tracks and the
        # baseline does not" is the claim worth defending.
        results["per_class_significance"] = {}
        for cls in names:
            idx = names.index(cls)
            rows = [i for i, t in enumerate(y_true) if t == idx]
            c_rec = results["clip"]["report"][cls]["recall"]
            f_rec = results["finetuned"]["report"][cls]["recall"]
            cls_ft, cls_clip, cls_p = mcnemar(
                [y_true[i] for i in rows],
                [ft_pred[i] for i in rows],
                [clip_pred[i] for i in rows],
            )
            mark = "*" if cls_p < 0.05 else " "
            print(f"  {cls:6s} recall  CLIP {c_rec:.2f} -> fine-tuned {f_rec:.2f} "
                  f" (n={len(rows):2d}, McNemar p={cls_p:.3f}){mark}")
            results["per_class_significance"][cls] = {
                "n": len(rows),
                "clip_recall": c_rec,
                "finetuned_recall": f_rec,
                "mcnemar_p": cls_p,
                "significant_at_05": bool(cls_p < 0.05),
            }
        print("  (* = significant at 0.05)")
        print("=" * 62)

        results["significance"] = {
            "n": n,
            "clip_ci": [clip_lo, clip_hi],
            "finetuned_ci": [ft_lo, ft_hi],
            "mcnemar_only_finetuned": only_ft,
            "mcnemar_only_clip": only_clip,
            "mcnemar_p": p_value,
            "significant_at_05": bool(p_value < 0.05),
        }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "comparison.json").write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwritten -> {MODEL_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
