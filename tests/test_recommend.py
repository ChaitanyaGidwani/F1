"""Unit tests for the rule-based recommendation engine."""

from __future__ import annotations

import itertools

import pytest

from app.labels import CLASS_DAMP, CLASS_DRY, CLASS_DRYING, CLASS_WET, CLASSES
from app.recommend import (
    ACTION_ACT,
    ACTION_HOLD,
    ACTION_MONITOR,
    ACTION_PREPARE,
    recommend,
)
from app.trend import (
    TREND_DRYING,
    TREND_STABLE,
    TREND_WETTING,
    FramePrediction,
    analyze,
)


def repeat(label: str, n: int, confidence: float = 0.9):
    return [FramePrediction.from_label(label, confidence) for _ in range(n)]


def test_stable_dry_needs_no_action():
    rec = recommend(analyze(repeat(CLASS_DRY, 12)))
    assert rec.action == ACTION_HOLD
    assert rec.urgency == 0
    assert "no action" in rec.message.lower()


def test_stable_wet_holds_strategy():
    rec = recommend(analyze(repeat(CLASS_WET, 12)))
    assert rec.action == ACTION_HOLD
    assert rec.tire == "Full Wet"


def test_drying_track_announces_tire_window():
    """The headline case from the brief."""
    result = analyze(repeat(CLASS_WET, 6) + repeat(CLASS_DAMP, 6))
    rec = recommend(result)
    assert result.trend == TREND_DRYING
    assert rec.action in (ACTION_PREPARE, ACTION_ACT)
    assert rec.urgency >= 2


def test_wetting_track_escalates_to_intermediates():
    result = analyze(repeat(CLASS_DRY, 6) + repeat(CLASS_DAMP, 6))
    rec = recommend(result)
    assert result.trend == TREND_WETTING
    assert "intermediate" in rec.tire.lower()


def test_insufficient_data_is_not_an_error():
    rec = recommend(analyze([]))
    assert rec.action == ACTION_MONITOR
    assert "collecting" in rec.message.lower()


def test_weather_hint_is_appended_not_authoritative():
    result = analyze(repeat(CLASS_WET, 12))
    rec = recommend(result, weather_hint="clear skies forecast")
    # Camera says wet, forecast says clear -> we still report wet.
    assert rec.tire == "Full Wet"
    assert "clear skies forecast" in rec.message


def test_every_trend_condition_pair_has_a_rule():
    """No (trend, class) combination may fall through to a KeyError in prod."""
    for trend, cls in itertools.product(
        [TREND_STABLE, TREND_DRYING, TREND_WETTING], CLASSES
    ):
        result = analyze(repeat(cls, 12))
        # Force the trend state to exercise the lookup table exhaustively.
        forced = type(result)(
            **{**result.__dict__, "trend": trend, "current_class": cls}
        )
        rec = recommend(forced)
        assert rec.message
        assert rec.action in (ACTION_HOLD, ACTION_MONITOR, ACTION_PREPARE, ACTION_ACT)
        assert 0 <= rec.urgency <= 3


def test_serialisation_shape():
    rec = recommend(analyze(repeat(CLASS_DRY, 12)))
    d = rec.to_dict()
    assert set(d) == {"message", "action", "urgency", "tire"}
