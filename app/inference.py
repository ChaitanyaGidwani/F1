"""Image -> class-probability inference, with a swappable backend.

Two implementations behind one interface:

* `FineTunedClassifier` - the primary path. Wraps
  `transformers.pipeline("image-classification", ...)` around the ViT we
  fine-tuned on the trackside dataset.
* `ClipZeroShotClassifier` - the fallback from the brief. No training required,
  noticeably weaker on this task (see docs/JUDGES.md), but it means the demo
  still classifies if the fine-tuned weights are missing.

`load_classifier()` picks the first that loads. The model is resolved and loaded
once at import/startup - there is no per-request network call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Protocol

import torch
from PIL import Image

from .clip_prompts import CLASS_PROMPTS
from .config import settings
from .labels import CLASSES, canonical

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Classifier(Protocol):
    name: str
    backend: str

    def predict(self, image: Image.Image) -> Dict[str, float]:
        """Return a probability over the four canonical classes."""


class FineTunedClassifier:
    """The fine-tuned ViT, loaded locally."""

    backend = "fine-tuned"

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import pipeline

        self.name = model_id
        self.device = device
        self.pipe = pipeline(
            "image-classification",
            model=model_id,
            device=device,
        )
        labels = list(self.pipe.model.config.id2label.values())
        # Fail loudly at startup rather than mislabelling silently at runtime.
        for label in labels:
            canonical(label)
        #: The classes this checkpoint can emit. Usually a subset of CLASSES,
        #: because Drying has no verified training data and is derived by the
        #: trend layer instead.
        self.model_classes = sorted(canonical(l) for l in labels)
        logger.info("loaded fine-tuned classifier %s on %s (%s)", model_id, device, labels)

    def predict(self, image: Image.Image) -> Dict[str, float]:
        raw = self.pipe(image, top_k=None)
        probs = {c: 0.0 for c in CLASSES}
        for item in raw:
            probs[canonical(item["label"])] += float(item["score"])
        total = sum(probs.values()) or 1.0
        return {c: v / total for c, v in probs.items()}


class ClipZeroShotClassifier:
    """Prompt-based fallback. Same prompt bank as the dataset triage step."""

    backend = "clip-zero-shot"

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_id
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        # Zero-shot can be prompted for every class, though it is poor at Damp.
        self.model_classes = list(CLASSES)

        self._prompts: List[str] = []
        self._owner: List[str] = []
        for cls in CLASSES:
            for prompt in CLASS_PROMPTS[cls]:
                self._prompts.append(prompt)
                self._owner.append(cls)

        with torch.no_grad():
            inputs = self.processor(
                text=self._prompts, return_tensors="pt", padding=True
            ).to(device)
            emb = self.model.get_text_features(**inputs)
            self._text_emb = emb / emb.norm(dim=-1, keepdim=True)
        logger.info("loaded CLIP zero-shot fallback %s on %s", model_id, device)

    def predict(self, image: Image.Image) -> Dict[str, float]:
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sims = (emb @ self._text_emb.T)[0] * 100.0

        per_class = []
        for cls in CLASSES:
            idx = [i for i, o in enumerate(self._owner) if o == cls]
            per_class.append(sims[idx].max())
        probs = torch.softmax(torch.stack(per_class), dim=0)
        return {c: float(p) for c, p in zip(CLASSES, probs)}


def load_classifier() -> Classifier:
    """Load the best available backend, preferring the fine-tuned model."""
    device = resolve_device(settings.device)

    if not settings.force_clip:
        model_id = settings.model_id
        is_local = Path(model_id).exists()
        if is_local or "/" in model_id:
            try:
                return FineTunedClassifier(model_id, device)
            except Exception as exc:  # noqa: BLE001 - fall back rather than crash
                logger.warning(
                    "fine-tuned model %r unavailable (%s); falling back to CLIP zero-shot",
                    model_id,
                    exc,
                )
    else:
        logger.info("WW_FORCE_CLIP=1 - using zero-shot fallback by request")

    return ClipZeroShotClassifier(settings.clip_model_id, device)
