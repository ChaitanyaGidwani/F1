"""Tests for video decoding and the upload guards.

The video here is generated in-process rather than checked in, so the suite has
no binary fixtures and still exercises a real decode.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module
from app.video import frames_from_video, is_video
from tests.test_api import ScriptedClassifier, make_image

av = pytest.importorskip("av", reason="video support needs PyAV")


def make_video(n_frames: int = 60, size=(160, 120)) -> bytes:
    """Encode a short clip that fades dark to light, one colour per frame."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"

    for i in range(n_frames):
        shade = int(255 * i / max(n_frames - 1, 1))
        img = Image.new("RGB", size, (shade, shade, shade))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


# ---------------------------------------------------------------- unit


def test_is_video_detects_by_mime_and_extension():
    assert is_video("clip.mp4", None)
    assert is_video("clip.MOV", None)
    assert is_video(None, "video/mp4")
    assert not is_video("frame.jpg", "image/jpeg")
    assert not is_video(None, None)


def test_frames_are_sampled_across_the_whole_clip():
    raw = make_video(60)
    frames = frames_from_video(raw, max_frames=6)
    assert 1 < len(frames) <= 6

    # The clip fades dark to light, so even sampling must show that spread.
    # Taking the first N frames instead would leave every sample nearly black.
    first = frames[0].convert("L").getpixel((80, 60))
    last = frames[-1].convert("L").getpixel((80, 60))
    assert last - first > 100, f"expected a wide spread, got {first} -> {last}"


def test_frame_cap_is_respected():
    frames = frames_from_video(make_video(60), max_frames=4)
    assert len(frames) <= 4


def test_garbage_bytes_raise_valueerror_not_a_crash():
    with pytest.raises(ValueError):
        frames_from_video(b"this is definitely not a video" * 100)


# ---------------------------------------------------------------- api


@pytest.fixture
def env(monkeypatch):
    fake = ScriptedClassifier()
    monkeypatch.setattr(main_module, "load_classifier", lambda: fake)
    with TestClient(main_module.app) as client:
        yield client, fake


def test_uploading_a_video_produces_a_sequence(env):
    client, fake = env
    from app.labels import CLASS_DAMP, CLASS_WET

    # 12 source frames decode 1:1, so the scripted wet-to-damp story lands
    # inside the 12-frame trend window rather than scrolling out of it.
    fake.set_script([CLASS_WET] * 6 + [CLASS_DAMP] * 6)
    res = client.post(
        "/api/sequence",
        files={"files": ("track.mp4", make_video(12), "video/mp4")},
        data={"session_id": "s-video", "reset": "true"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    # One uploaded file expanded into many analysed frames.
    assert body["frames"] == 12
    assert body["steps"][0]["filename"].startswith("track.mp4 #")
    assert body["trend"]["trend"] == "DRYING"


def test_long_video_is_subsampled_not_truncated(env):
    """A long clip must be covered end to end, capped, not read front-first."""
    client, fake = env
    fake.set_script(["Wet"])
    res = client.post(
        "/api/sequence",
        files={"files": ("long.mp4", make_video(120), "video/mp4")},
        data={"session_id": "s-long", "reset": "true"},
    )
    assert res.status_code == 200, res.text
    frames = res.json()["frames"]
    assert 1 < frames <= main_module.settings.video_max_frames


def test_corrupt_video_is_a_400_not_a_500(env):
    client, _ = env
    res = client.post(
        "/api/sequence",
        files={"files": ("broken.mp4", b"not a video at all", "video/mp4")},
        data={"session_id": "s-badvideo"},
    )
    assert res.status_code == 400
    assert "error" in res.json()


def test_too_many_files_is_rejected(env):
    client, _ = env
    limit = main_module.settings.max_frames
    files = [
        ("files", (f"f{i:03d}.jpg", make_image(), "image/jpeg"))
        for i in range(limit + 5)
    ]
    res = client.post("/api/sequence", files=files, data={"session_id": "s-many"})
    assert res.status_code == 413
    assert "limit" in res.json()["error"].lower()


def test_decompression_bomb_is_a_400_not_a_500(env, monkeypatch):
    """A pixel bomb must be refused cleanly rather than escaping as a 500."""
    client, _ = env

    def explode(*_args, **_kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(main_module.Image, "open", explode)
    res = client.post(
        "/api/predict",
        files={"file": ("bomb.png", make_image(), "image/png")},
        data={"session_id": "s-bomb"},
    )
    assert res.status_code == 400


def test_health_reports_video_and_model_classes(env):
    client, _ = env
    body = client.get("/api/health").json()
    assert body["video_supported"] is True
    assert body["max_frames"] == main_module.settings.max_frames
    assert "model_classes" in body
