"""Unit tests for trend detection, run against synthetic prediction sequences.

`analyze` is a pure function, so no model is needed to test it.
"""

from __future__ import annotations

import pytest

from app.labels import CLASS_DAMP, CLASS_DRY, CLASS_DRYING, CLASS_WET
from app.trend import (
    TREND_DRYING,
    TREND_INSUFFICIENT,
    TREND_STABLE,
    TREND_WETTING,
    FramePrediction,
    TrackConditionTracker,
    TrendConfig,
    analyze,
)


def seq(*labels: str, confidence: float = 0.9):
    """Build a frame sequence from label names."""
    return [FramePrediction.from_label(l, confidence) for l in labels]


def repeat(label: str, n: int, confidence: float = 0.9):
    return seq(*([label] * n), confidence=confidence)


# --------------------------------------------------------------------------
# Frame-level plumbing
# --------------------------------------------------------------------------


def test_from_probs_normalises_and_picks_top():
    f = FramePrediction.from_probs({"Dry": 1.0, "Wet": 3.0})
    assert f.label == CLASS_WET
    assert f.confidence == pytest.approx(0.75)
    assert sum(f.probs.values()) == pytest.approx(1.0)


def test_from_label_spreads_remaining_mass():
    f = FramePrediction.from_label(CLASS_DRY, 0.7)
    assert f.label == CLASS_DRY
    assert f.confidence == pytest.approx(0.7)
    assert f.probs[CLASS_WET] == pytest.approx(0.1)


def test_wetness_projection_is_confidence_aware():
    confident = FramePrediction.from_label(CLASS_WET, 1.0)
    unsure = FramePrediction.from_label(CLASS_WET, 0.4)
    assert confident.wetness == pytest.approx(2.0)
    # An unsure "Wet" call should sit lower on the wetness axis than a certain one.
    assert unsure.wetness < confident.wetness


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        FramePrediction.from_label("Soaking")


def test_zero_probability_vector_rejected():
    with pytest.raises(ValueError):
        FramePrediction.from_probs({"Dry": 0.0, "Wet": 0.0})


# --------------------------------------------------------------------------
# Trend states
# --------------------------------------------------------------------------


def test_empty_buffer_is_insufficient():
    r = analyze([])
    assert r.trend == TREND_INSUFFICIENT
    assert r.frames == 0
    assert r.wetness_history == []


def test_too_few_frames_is_insufficient():
    r = analyze(repeat(CLASS_WET, 3))
    assert r.trend == TREND_INSUFFICIENT
    assert r.frames == 3
    # It should still report a current condition even without a trend.
    assert r.current_class == CLASS_WET


def test_steady_wet_is_stable():
    r = analyze(repeat(CLASS_WET, 12))
    assert r.trend == TREND_STABLE
    assert r.current_class == CLASS_WET
    assert r.transition is None
    assert r.trend_strength == 0.0


def test_steady_dry_is_stable():
    r = analyze(repeat(CLASS_DRY, 12))
    assert r.trend == TREND_STABLE
    assert r.current_class == CLASS_DRY
    assert r.wetness == pytest.approx(0.0, abs=0.2)


def test_wet_to_dry_sequence_flags_drying():
    frames = repeat(CLASS_WET, 4) + repeat(CLASS_DAMP, 4) + repeat(CLASS_DRY, 4)
    r = analyze(frames)
    assert r.trend == TREND_DRYING
    assert r.slope < 0
    assert 0.0 < r.trend_strength <= 1.0


def test_dry_to_wet_sequence_flags_wetting():
    frames = repeat(CLASS_DRY, 4) + repeat(CLASS_DAMP, 4) + repeat(CLASS_WET, 4)
    r = analyze(frames)
    assert r.trend == TREND_WETTING
    assert r.slope > 0


def test_wet_to_damp_reports_transition_label():
    """The brief's explicit case: dominant class shifting Wet -> Damp."""
    frames = repeat(CLASS_WET, 6) + repeat(CLASS_DAMP, 6)
    r = analyze(frames)
    assert r.trend == TREND_DRYING
    assert r.transition == f"{CLASS_WET}->{CLASS_DAMP}"


def test_damp_to_dry_reports_transition_label():
    frames = repeat(CLASS_DAMP, 6) + repeat(CLASS_DRY, 6)
    r = analyze(frames)
    assert r.trend == TREND_DRYING
    assert r.transition == f"{CLASS_DAMP}->{CLASS_DRY}"


def test_noisy_oscillation_is_not_a_trend():
    """A classifier flickering between adjacent classes must not read as weather."""
    frames = seq(*([CLASS_DAMP, CLASS_DRYING] * 6))
    r = analyze(frames)
    assert r.trend == TREND_STABLE
    assert r.transition is None


def test_single_outlier_frame_does_not_trigger_a_trend():
    frames = repeat(CLASS_WET, 11) + seq(CLASS_DRY)
    r = analyze(frames)
    assert r.trend == TREND_STABLE


def test_low_confidence_drift_is_ignored():
    """Slope must clear the threshold; a barely-moving signal stays STABLE."""
    frames = [
        FramePrediction.from_probs({CLASS_DAMP: 0.5 + i * 0.001, CLASS_DRYING: 0.5 - i * 0.001})
        for i in range(12)
    ]
    r = analyze(frames)
    assert r.trend == TREND_STABLE


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_only_last_window_frames_are_considered():
    """Old wet frames outside the window must not drag the trend."""
    frames = repeat(CLASS_WET, 8) + repeat(CLASS_DRY, 12)
    r = analyze(frames, TrendConfig(window=12))
    assert r.frames == 12
    assert r.current_class == CLASS_DRY
    assert r.trend == TREND_STABLE


def test_custom_window_changes_verdict():
    frames = repeat(CLASS_WET, 6) + repeat(CLASS_DRY, 6)
    wide = analyze(frames, TrendConfig(window=12))
    narrow = analyze(frames, TrendConfig(window=4))
    assert wide.trend == TREND_DRYING
    assert narrow.trend == TREND_STABLE


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        TrendConfig(window=1)
    with pytest.raises(ValueError):
        TrendConfig(ema_alpha=0.0)


# --------------------------------------------------------------------------
# Stateful tracker
# --------------------------------------------------------------------------


def test_tracker_accumulates_and_detects_drying():
    tracker = TrackConditionTracker()
    result = None
    for label in [CLASS_WET] * 5 + [CLASS_DAMP] * 4 + [CLASS_DRY] * 3:
        result = tracker.update(FramePrediction.from_label(label, 0.9))
    assert result is not None
    assert result.trend == TREND_DRYING
    assert len(tracker) == 12


def test_tracker_respects_maxlen():
    tracker = TrackConditionTracker(TrendConfig(window=5))
    for _ in range(20):
        tracker.update(FramePrediction.from_label(CLASS_DRY))
    assert len(tracker) == 5


def test_tracker_reset_clears_history():
    tracker = TrackConditionTracker()
    for _ in range(6):
        tracker.update(FramePrediction.from_label(CLASS_WET))
    tracker.reset()
    assert len(tracker) == 0
    assert tracker.analyze().trend == TREND_INSUFFICIENT


def test_result_serialises_cleanly():
    r = analyze(repeat(CLASS_WET, 6) + repeat(CLASS_DRY, 6))
    d = r.to_dict()
    assert set(d) >= {"current_class", "trend", "wetness_history", "slope", "transition"}
    assert len(d["wetness_history"]) == 12
    assert all(isinstance(x, float) for x in d["wetness_history"])
