"""Runtime settings, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "vit-track-condition"


def _env(name: str, default: str) -> str:
    # An empty variable counts as unset. Container runtimes set env vars to ""
    # for unfilled build args, and an empty WW_MODEL_ID would otherwise resolve
    # to Path("") -> "." and send the loader looking for a model in the cwd.
    return os.environ.get(name, "").strip() or default


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
    max_video_bytes: int = int(_env("WW_MAX_VIDEO_BYTES", str(64 * 1024 * 1024)))
    max_sessions: int = int(_env("WW_MAX_SESSIONS", "64"))
    #: Ceiling on frames analysed per request. Without it, one upload of a few
    #: hundred frames occupies a worker for the length of the demo.
    max_frames: int = int(_env("WW_MAX_FRAMES", "60"))
    #: Frames sampled from an uploaded video, spread across the whole clip.
    video_max_frames: int = int(_env("WW_VIDEO_MAX_FRAMES", "36"))

    #: Comma-separated allowed origins. Empty by default: the backend serves the
    #: frontend itself, so the demo is same-origin and needs no CORS at all.
    #: Set this only when a browser app on another origin has to call the API.
    cors_origins: str = _env("WW_CORS_ORIGINS", "")

    @property
    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
