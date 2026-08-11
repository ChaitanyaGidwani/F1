"""End-to-end API tests against a scripted classifier.

The real model is not loaded here. Swapping in a classifier whose
outputs we choose lets us assert that a *specific* weather story produces a
*specific* strategy call - which is the behaviour that matters and which a real
model's noise would make untestable.
"""

from __future__ import annotations

import io
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module
from app.labels import CLASS_DAMP, CLASS_DRY, CLASS_WET, CLASSES


def one_hot(label: str, confidence: float = 0.92) -> Dict[str, float]:
    rest = (1.0 - confidence) / (len(CLASSES) - 1)
    return {c: (confidence if c == label else rest) for c in CLASSES}


class ScriptedClassifier:
    """Returns a predetermined label per call, so sequences are reproducible."""

    backend = "fine-tuned"
    name = "scripted-test-model"

    def __init__(self) -> None:
        self.script: List[str] = [CLASS_DRY]
        self.calls = 0

    def set_script(self, labels: List[str]) -> None:
        self.script = labels
        self.calls = 0

    def predict(self, image) -> Dict[str, float]:
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return one_hot(self.script[index])


def make_image(color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def env(monkeypatch):
    fake = ScriptedClassifier()
    monkeypatch.setattr(main_module, "load_classifier", lambda: fake)
    with TestClient(main_module.app) as client:
        yield client, fake


def upload(client, session_id: str, n: int = 1):
    files = [("files", (f"f{i:02d}.jpg", make_image(), "image/jpeg")) for i in range(n)]
    return client.post(
        "/api/sequence",
        files=files,
        data={"session_id": session_id, "reset": "true"},
    )


# ---------------------------------------------------------------- health


def test_health_reports_backend(env):
    client, _ = env
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "fine-tuned"
    assert body["classes"] == CLASSES
    assert set(body["class_colors"]) == set(CLASSES)


# ---------------------------------------------------------------- predict


def test_single_predict_shape(env):
    client, fake = env
    fake.set_script([CLASS_WET])
    res = client.post(
        "/api/predict",
        files={"file": ("frame.jpg", make_image(), "image/jpeg")},
        data={"session_id": "s-single"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["frame"]["label"] == CLASS_WET
    assert 0 < body["frame"]["confidence"] <= 1
    assert set(body["frame"]["probs"]) == set(CLASSES)
    assert body["trend"]["trend"] == "INSUFFICIENT_DATA"  # one frame is not a trend
    assert body["recommendation"]["message"]


def test_repeated_predicts_accumulate_a_trend(env):
    """Uploading frames one at a time must build the same trend as a batch."""
    client, fake = env
    fake.set_script([CLASS_WET] * 6 + [CLASS_DAMP] * 6)
    body = None
    for _ in range(12):
        body = client.post(
            "/api/predict",
            files={"file": ("frame.jpg", make_image(), "image/jpeg")},
            data={"session_id": "s-accumulate"},
        ).json()
    assert body["trend"]["trend"] == "DRYING"
    assert body["trend"]["frames"] == 12


def test_rejects_non_image(env):
    client, _ = env
    res = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
        data={"session_id": "s-bad"},
    )
    assert res.status_code == 400
    assert "error" in res.json()


# ---------------------------------------------------------------- sequence


def test_sequence_returns_per_frame_trace(env):
    client, fake = env
    fake.set_script([CLASS_WET] * 6 + [CLASS_DAMP] * 6)
    body = upload(client, "s-seq", 12).json()

    assert body["frames"] == 12
    assert len(body["steps"]) == 12
    # The trend should be absent early and present once enough frames exist.
    assert body["steps"][0]["trend"]["trend"] == "INSUFFICIENT_DATA"
    assert body["trend"]["trend"] == "DRYING"


def test_drying_sequence_produces_tire_call(env):
    """The headline demo: a wet track that dries must open the tyre window."""
    client, fake = env
    fake.set_script([CLASS_WET] * 6 + [CLASS_DAMP] * 6)
    body = upload(client, "s-drying", 12).json()

    trend = body["trend"]
    rec = body["recommendation"]
    assert trend["trend"] == "DRYING"
    assert trend["transition"] == f"{CLASS_WET}->{CLASS_DAMP}"
    assert rec["urgency"] >= 2
    assert rec["action"] in ("PREPARE", "ACT")


def test_wetting_sequence_escalates(env):
    client, fake = env
    fake.set_script([CLASS_DRY] * 6 + [CLASS_DAMP] * 6)
    body = upload(client, "s-wetting", 12).json()
    assert body["trend"]["trend"] == "WETTING"
    assert "ntermediate" in body["recommendation"]["tire"]


def test_stable_dry_sequence_recommends_no_action(env):
    client, fake = env
    fake.set_script([CLASS_DRY] * 12)
    body = upload(client, "s-stable", 12).json()
    assert body["trend"]["trend"] == "STABLE"
    assert body["recommendation"]["action"] == "HOLD"
    assert body["recommendation"]["urgency"] == 0


def test_empty_sequence_rejected(env):
    client, _ = env
    res = client.post("/api/sequence", data={"session_id": "s-empty"})
    assert res.status_code in (400, 422)


def test_weather_hint_is_echoed_in_message(env):
    client, fake = env
    fake.set_script([CLASS_WET] * 12)
    res = client.post(
        "/api/sequence",
        files=[("files", (f"f{i}.jpg", make_image(), "image/jpeg")) for i in range(12)],
        data={"session_id": "s-hint", "weather_hint": "radar clear in 5"},
    )
    assert "radar clear in 5" in res.json()["recommendation"]["message"]


# ---------------------------------------------------------------- sessions


def test_sessions_are_isolated(env):
    client, fake = env
    fake.set_script([CLASS_WET] * 12)
    upload(client, "s-a", 12)

    fake.set_script([CLASS_DRY] * 12)
    body_b = upload(client, "s-b", 12).json()

    # Session B must not inherit session A's wet history.
    assert body_b["trend"]["current_class"] == CLASS_DRY
    assert body_b["trend"]["trend"] == "STABLE"


def test_reset_clears_the_buffer(env):
    client, fake = env
    fake.set_script([CLASS_WET] * 12)
    upload(client, "s-reset", 12)

    client.post("/api/session/reset", data={"session_id": "s-reset"})

    fake.set_script([CLASS_DRY])
    body = client.post(
        "/api/predict",
        files={"file": ("frame.jpg", make_image(), "image/jpeg")},
        data={"session_id": "s-reset"},
    ).json()
    assert body["trend"]["frames"] == 1


# ---------------------------------------------------------------- frontend


def test_frontend_is_served(env):
    client, _ = env
    res = client.get("/")
    assert res.status_code == 200
    assert "Weather Whiplash" in res.text
