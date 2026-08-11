"""Runtime settings, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "vit-track-condition"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    #: Local path or Hub id of the fine-tuned classifier. Loaded at startup -
    #: never called over the network per request, so a flaky conference wifi
    #: cannot take the demo down mid-presentation.
    model_id: str = _env("WW_MODEL_ID", str(DEFAULT_MODEL_DIR))
    #: Zero-shot fallback, used only if the fine-tuned model cannot be loaded.
    clip_model_id: str = _env("WW_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    #: "auto" resolves to mps -> cuda -> cpu.
    device: str = _env("WW_DEVICE", "auto")
    #: Force the CLIP fallback even when the fine-tuned model is present.
    force_clip: bool = _env("WW_FORCE_CLIP", "0") == "1"

    window: int = int(_env("WW_WINDOW", "12"))
    min_frames: int = int(_env("WW_MIN_FRAMES", "4"))
    slope_threshold: float = float(_env("WW_SLOPE_THRESHOLD", "0.025"))
    min_net_delta: float = float(_env("WW_MIN_NET_DELTA", "0.15"))
    ema_alpha: float = float(_env("WW_EMA_ALPHA", "0.4"))

    max_upload_bytes: int = int(_env("WW_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
    max_sessions: int = int(_env("WW_MAX_SESSIONS", "64"))


settings = Settings()
