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
    forecast_signal,
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


# ---------------------------------------------------------------- forecast


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("radar shows rain in 10 minutes", "rain"),
        ("heavy showers approaching", "rain"),
        ("thunderstorm over sector 2", "rain"),
        ("clear skies for the next hour", "clear"),
        ("no rain expected", "clear"),  # contains "rain" but means the opposite
        ("track temp 41C", None),
        ("", None),
        (None, None),
    ],
)
def test_forecast_signal_reading(hint, expected):
    assert forecast_signal(hint) == expected


def test_rain_forecast_on_a_dry_track_raises_the_alert():
    """The case the tool exists to catch: forecast and camera disagree."""
    result = analyze(repeat(CLASS_DRY, 12))
    calm = recommend(result)
    warned = recommend(result, weather_hint="rain in 10 minutes")

    assert calm.action == ACTION_HOLD
    assert warned.urgency > calm.urgency
    assert "prepare early" in warned.message
    # The camera still decides the condition and the tyre.
    assert warned.tire == calm.tire


def test_rain_forecast_agreeing_with_a_wet_track_changes_nothing():
    result = analyze(repeat(CLASS_WET, 12))
    plain = recommend(result)
    noted = recommend(result, weather_hint="more rain coming")
    assert noted.action == plain.action
    assert noted.urgency == plain.urgency


def test_forecast_alone_never_demands_a_pit_stop():
    """A typed note may raise the alert, but not to ACT."""
    result = analyze(repeat(CLASS_DAMP, 6) + repeat(CLASS_DRY, 6))
    warned = recommend(result, weather_hint="storm approaching")
    assert warned.action != ACTION_ACT or recommend(result).action == ACTION_ACT


def test_irrelevant_note_does_not_escalate():
    result = analyze(repeat(CLASS_DRY, 12))
    plain = recommend(result)
    noted = recommend(result, weather_hint="track temp 41C")
    assert noted.action == plain.action
    assert "track temp 41C" in noted.message


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
