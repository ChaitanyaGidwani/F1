"""Fine-tune a ViT backbone on the trackside condition dataset.

Backbone: google/vit-base-patch16-224-in21k (swap with --backbone if live
inference speed becomes the constraint).

A note on augmentation, because it is the one place a generic recipe actively
hurts here: the visual signal separating Dry / Damp / Wet *is* surface luminance
and specular reflection. The usual ColorJitter(brightness, saturation, contrast)
that helps on ImageNet-style tasks would erase the cue being learned and
teach the model to guess from sky and car colour instead. So augmentation is
restricted to geometry (crop, flip, small rotation) plus a very mild jitter that
stays well inside the class boundary.

Usage:
    python -m training.train_vit --epochs 8
    python -m training.train_vit --push-to-hub --hub-model-id <org>/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torchvision.transforms import (
    Compose,
    ColorJitter,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset"
OUT_DIR = ROOT / "models" / "vit-track-condition"

DEFAULT_BACKBONE = "google/vit-base-patch16-224-in21k"


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted loss.

    Damp is roughly a tenth as common as Wet in the collected data, because
    photographers shoot dramatic conditions and a merely damp track is not
    dramatic. Unweighted, the model gets ~95% accuracy by never predicting Damp
    at all, which is useless: Damp is the state that decides whether a
    car is on intermediates.
    """

    def __init__(self, class_weights=None, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(
            outputs.logits, labels, weight=weight, label_smoothing=self.label_smoothing
        )
        return (loss, outputs) if return_outputs else loss


def build_transforms(processor):
    mean, std = processor.image_mean, processor.image_std
    size = processor.size.get("height", 224)
    normalize = Normalize(mean=mean, std=std)

    train_tf = Compose(
        [
            # Aspect ratio is left wide so crops stay closer to the squashed
            # full frame the model sees at inference time.
            RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.75, 1.4)),
            RandomHorizontalFlip(),
            RandomRotation(7),
            # Mild only: see module docstring.
            ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.01),
            ToTensor(),
            normalize,
        ]
    )
    # Match what the deployed `pipeline` does: ViTImageProcessor resizes the
    # whole image to size x size. Resize-shortest-side plus CenterCrop would
    # throw away the left and right of a wide trackside shot, so eval here would
    # measure a preprocessing that never runs in production - and, because the
    # best checkpoint is chosen on this metric, would select for the wrong thing.
    eval_tf = Compose([Resize((size, size)), ToTensor(), normalize])
    return train_tf, eval_tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--no-class-weights", dest="class_weights",
                        action="store_false", default=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = load_dataset("imagefolder", data_dir=args.data_dir)
    print({k: len(v) for k, v in ds.items()})

    # imagefolder infers labels from folder names; pin them to our canonical order.
    folder_names = ds["train"].features["label"].names
    label2id = {name: i for i, name in enumerate(folder_names)}
    id2label = {i: name for name, i in label2id.items()}
    print("labels:", id2label)

    processor = AutoImageProcessor.from_pretrained(args.backbone)
    train_tf, eval_tf = build_transforms(processor)

    def apply_train(batch):
        batch["pixel_values"] = [train_tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    def apply_eval(batch):
        batch["pixel_values"] = [eval_tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    ds["train"].set_transform(apply_train)
    for split in ("validation", "test"):
        if split in ds:
            ds[split].set_transform(apply_eval)

    model = AutoModelForImageClassification.from_pretrained(
        args.backbone,
        num_labels=len(folder_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def collate(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["label"] for e in examples], dtype=torch.long),
        }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dataloader_num_workers=0,
    )

    class_weights = None
    if args.class_weights:
        counts = [0] * len(folder_names)
        for label in ds["train"]["label"]:
            counts[label] += 1
        total = sum(counts)
        # Inverse frequency, normalised so the mean weight is 1.
        raw = [total / (len(counts) * c) if c else 0.0 for c in counts]
        scale = len(raw) / sum(raw)
        class_weights = torch.tensor([w * scale for w in raw], dtype=torch.float)
        print("class counts:", dict(zip(folder_names, counts)))
        print("class weights:", {n: round(float(w), 3)
                                 for n, w in zip(folder_names, class_weights)})

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = {}
    for split in ("validation", "test"):
        if split not in ds:
            continue
        metrics = trainer.evaluate(ds[split], metric_key_prefix=split)
        results[split] = metrics
        print(f"\n=== {split} ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        preds = trainer.predict(ds[split])
        y_pred = np.argmax(preds.predictions, axis=-1)
        y_true = preds.label_ids
        names = [id2label[i] for i in range(len(id2label))]
        print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
        results[f"{split}_report"] = classification_report(
            y_true, y_pred, target_names=names, zero_division=0, output_dict=True
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "eval_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nmodel saved -> {out}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"pushed -> {args.hub_model_id}")


if __name__ == "__main__":
    main()
