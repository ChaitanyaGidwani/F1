"""Triage the raw Commons pool with CLIP zero-shot, ahead of human review.

This is a labelling aid, not the label source. CLIP sorts the pool into four
candidate buckets plus a reject bucket (portraits, garage shots, crowds, and
anything else where the racing surface isn't the subject), which makes a manual
pass over the pool feasible.

Keeping that distinction matters. If CLIP's guesses became the ground truth,
fine-tuning would just distil CLIP and the accuracy number would be circular.
Labels are confirmed or corrected by eye in make_contactsheets.py, and the test
split is hand-verified. The resulting disagreement rate is reported in
docs/JUDGES.md.

Usage:
    python -m data_pipeline.presort_clip --batch-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.clip_prompts import CLASS_PROMPTS, REJECT, REJECT_PROMPTS
from app.labels import CLASSES

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
PRESORT = ROOT / "data" / "raw" / "presort.jsonl"

CLIP_MODEL = "openai/clip-vit-base-patch32"

# Prompt bank lives in app/clip_prompts.py so the triage step and the runtime
# fallback classifier can never drift apart.
PROMPTS: Dict[str, List[str]] = dict(CLASS_PROMPTS)
PROMPTS[REJECT] = REJECT_PROMPTS


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"triaging {len(records)} images")

    device = pick_device()
    print(f"device: {device}")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Flatten prompts, remembering which class each belongs to.
    flat: List[str] = []
    owner: List[str] = []
    for cls, prompts in PROMPTS.items():
        for p in prompts:
            flat.append(p)
            owner.append(cls)

    with torch.no_grad():
        text_inputs = processor(text=flat, return_tensors="pt", padding=True).to(device)
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    all_classes = list(PROMPTS.keys())
    out = PRESORT.open("w")
    written = 0

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        images, keep = [], []
        for rec in batch:
            path = ROOT / rec["path"]
            try:
                images.append(Image.open(path).convert("RGB"))
                keep.append(rec)
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ text_emb.T) * 100.0  # [B, n_prompts]

        for row, rec in zip(sims, keep):
            # Best prompt per class, then softmax across classes.
            per_class = {}
            for cls in all_classes:
                idx = [i for i, o in enumerate(owner) if o == cls]
                per_class[cls] = float(row[idx].max())
            logits = torch.tensor([per_class[c] for c in all_classes])
            probs = torch.softmax(logits, dim=0)
            scores = {c: round(float(p), 4) for c, p in zip(all_classes, probs)}
            best = max(scores, key=lambda c: scores[c])

            out.write(
                json.dumps(
                    {
                        "sha1": rec["sha1"],
                        "path": rec["path"],
                        "hint": rec.get("hint"),
                        "category": rec.get("category"),
                        "clip_label": best,
                        "clip_confidence": scores[best],
                        "clip_scores": scores,
                        "is_reject": best == REJECT,
                    }
                )
                + "\n"
            )
            written += 1

        print(f"  {min(start + args.batch_size, len(records))}/{len(records)}")

    out.close()

    counts: Dict[str, int] = {}
    for line in PRESORT.read_text().splitlines():
        if line.strip():
            counts[json.loads(line)["clip_label"]] = counts.get(json.loads(line)["clip_label"], 0) + 1
    print(f"\nwrote {written} records -> {PRESORT}")
    for cls in CLASSES + [REJECT]:
        print(f"  {cls:10s} {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()
