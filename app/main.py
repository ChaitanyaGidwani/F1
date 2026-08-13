"""FastAPI backend: image in, condition + trend + recommendation out.

The model is loaded once at startup and stays in-process. Nothing calls the
hosted Inference API, so serving a request needs no network access.

Endpoints:
    GET  /api/health            backend, device, model id, class list
    POST /api/predict           one frame -> condition + trend + recommendation
    POST /api/sequence          N frames in order -> per-frame trace + verdict
    POST /api/session/reset     clear a session's rolling buffer
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import settings
from .inference import Classifier, load_classifier, resolve_device
from .labels import CLASS_COLORS, CLASSES, WETNESS
from .recommend import recommend
from .trend import FramePrediction, TrackConditionTracker, TrendConfig
from .video import frames_from_video, is_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("weather-whiplash")

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
DEMO_DIR = ROOT_DIR / "data" / "demo"

try:  # video decoding is optional; the app still serves images without it
    import av  # noqa: F401

    VIDEO_SUPPORTED = True
except ImportError:  # pragma: no cover
    VIDEO_SUPPORTED = False

TREND_CONFIG = TrendConfig(
    window=settings.window,
    min_frames=settings.min_frames,
    slope_threshold=settings.slope_threshold,
    min_net_delta=settings.min_net_delta,
    ema_alpha=settings.ema_alpha,
)


class SessionStore:
    """Per-camera rolling buffers, LRU-capped so a long demo cannot leak memory."""

    def __init__(self, config: TrendConfig, max_sessions: int) -> None:
        self._config = config
        self._max = max_sessions
        self._lock = Lock()
        self._sessions: "OrderedDict[str, TrackConditionTracker]" = OrderedDict()

    def get(self, session_id: str) -> TrackConditionTracker:
        with self._lock:
            tracker = self._sessions.get(session_id)
            if tracker is None:
                tracker = TrackConditionTracker(self._config)
                self._sessions[session_id] = tracker
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
            return tracker

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


state: Dict[str, object] = {}
sessions = SessionStore(TREND_CONFIG, settings.max_sessions)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    classifier = load_classifier()
    # Warm the graph so the first real request is not the slow one.
    try:
        classifier.predict(Image.new("RGB", (224, 224), (128, 128, 128)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("warmup failed: %s", exc)
    state["classifier"] = classifier
    logger.info("ready in %.1fs (backend=%s)", time.time() - started, classifier.backend)
    yield
    state.clear()


app = FastAPI(title="Weather Whiplash", version="1.0.0", lifespan=lifespan)
# The backend serves the frontend, so the demo is same-origin and needs no CORS.
# A wildcard would let any page on the internet drive this API, so it is opt-in
# through WW_CORS_ORIGINS rather than the default.
if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_classifier() -> Classifier:
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    return classifier  # type: ignore[return-value]


def read_image(raw: bytes, filename: str) -> Image.Image:
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"{filename} exceeds upload limit")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    # DecompressionBombError is not an OSError, so without naming it here a
    # pixel-bomb image escapes as a 500 instead of a clean rejection.
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail=f"{filename} is not a readable image")


def expand_uploads(uploads: List[UploadFile]) -> List[Tuple[str, Image.Image]]:
    """Turn uploads into one ordered list of frames.

    A video expands into many frames, an image contributes one, and both end up
    in the same sequence analysis. The frame ceiling matters: without it a
    single upload of a few hundred images ties up a worker for minutes.
    """
    if len(uploads) > settings.max_frames:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(uploads)}, limit is {settings.max_frames}",
        )

    frames: List[Tuple[str, Image.Image]] = []
    for index, upload in enumerate(uploads):
        name = upload.filename or f"frame-{index}"
        raw = upload.file.read()

        if is_video(upload.filename, upload.content_type):
            if len(raw) > settings.max_video_bytes:
                raise HTTPException(
                    status_code=413, detail=f"{name} exceeds the video size limit"
                )
            try:
                decoded = frames_from_video(
                    raw, max_frames=settings.video_max_frames
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{name}: {exc}")
            logger.info("decoded %d frames from %s", len(decoded), name)
            frames.extend((f"{name} #{i + 1}", img) for i, img in enumerate(decoded))
        else:
            frames.append((name, read_image(raw, name)))

        if len(frames) > settings.max_frames:
            raise HTTPException(
                status_code=413,
                detail=f"Too many frames: limit is {settings.max_frames}",
            )
    return frames


def classify(image: Image.Image) -> Dict[str, object]:
    classifier = get_classifier()
    t0 = time.perf_counter()
    probs = classifier.predict(image)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    prediction = FramePrediction.from_probs(probs)
    return {"prediction": prediction, "probs": probs, "latency_ms": latency_ms}


def frame_payload(prediction: FramePrediction, probs: Dict[str, float], latency_ms: float):
    return {
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "wetness": round(prediction.wetness, 4),
        "probs": {c: round(float(probs[c]), 4) for c in CLASSES},
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/api/health")
def health() -> Dict[str, object]:
    classifier = state.get("classifier")
    return {
        "status": "ok" if classifier else "loading",
        "backend": getattr(classifier, "backend", None),
        "model": getattr(classifier, "name", None),
        "device": resolve_device(settings.device),
        "classes": CLASSES,
        "class_colors": CLASS_COLORS,
        "wetness_scale": WETNESS,
        "window": settings.window,
        "max_frames": settings.max_frames,
        "video_supported": VIDEO_SUPPORTED,
        # What the classifier can actually emit, which is not the same as the
        # states the system reports: Drying comes from the trend layer. The UI
        # reads this so the legend can say so rather than implying the model
        # predicts a class it was never trained on.
        "model_classes": getattr(classifier, "model_classes", None),
    }


@app.post("/api/predict")
def predict(
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
    weather_hint: Optional[str] = Form(default=None),
) -> Dict[str, object]:
    """Classify one frame and fold it into the session's rolling trend."""
    session_id = session_id or uuid.uuid4().hex
    image = read_image(file.file.read(), file.filename or "upload")

    result = classify(image)
    tracker = sessions.get(session_id)
    trend = tracker.update(result["prediction"])  # type: ignore[arg-type]
    rec = recommend(trend, weather_hint=weather_hint)

    return {
        "session_id": session_id,
        "frame": frame_payload(result["prediction"], result["probs"], result["latency_ms"]),  # type: ignore[arg-type]
        "trend": trend.to_dict(),
        "recommendation": rec.to_dict(),
    }


@app.post("/api/sequence")
def sequence(
    files: List[UploadFile] = File(...),
    session_id: str = Form(default=""),
    weather_hint: Optional[str] = Form(default=None),
    reset: bool = Form(default=True),
) -> Dict[str, object]:
    """Classify an ordered burst of frames - the 'drying track' demo path.

    Accepts images, a video, or a mix. A video is decoded into frames sampled
    across the whole clip, so an mp4 and a folder of stills take the same path.

    Frames are processed in the order given, and the trend after *each* frame is
    returned, so the UI can replay how the verdict evolved rather than only
    showing the final state.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No frames uploaded")

    frames = expand_uploads(files)

    session_id = session_id or uuid.uuid4().hex
    if reset:
        sessions.reset(session_id)
    tracker = sessions.get(session_id)

    steps = []
    trend = None
    rec = None
    for index, (name, image) in enumerate(frames):
        result = classify(image)
        trend = tracker.update(result["prediction"])  # type: ignore[arg-type]
        rec = recommend(trend, weather_hint=weather_hint)
        steps.append(
            {
                "index": index,
                "filename": name,
                "frame": frame_payload(
                    result["prediction"], result["probs"], result["latency_ms"]  # type: ignore[arg-type]
                ),
                "trend": trend.to_dict(),
                "recommendation": rec.to_dict(),
            }
        )

    return {
        "session_id": session_id,
        "frames": len(steps),
        "steps": steps,
        "trend": trend.to_dict() if trend else None,
        "recommendation": rec.to_dict() if rec else None,
    }


@app.post("/api/session/reset")
def reset_session(session_id: str = Form(...)) -> Dict[str, object]:
    sessions.reset(session_id)
    return {"session_id": session_id, "status": "reset"}


@app.get("/api/demo-sequence")
def demo_sequence() -> Dict[str, object]:
    """Ordered frames of a bundled wet -> drying -> dry transition.

    Judges get a one-click demo that does not depend on them having track
    footage to hand. Files are numbered, so lexical order is temporal order.
    """
    if not DEMO_DIR.exists():
        raise HTTPException(status_code=404, detail="No demo sequence bundled")
    files = sorted(
        p.name for p in DEMO_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not files:
        raise HTTPException(status_code=404, detail="No demo sequence bundled")
    return {"files": [f"/demo/{name}" for name in files], "count": len(files)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# Mount order matters: the catch-all frontend mount must come last.
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
