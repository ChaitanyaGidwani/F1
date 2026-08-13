"""Rule-based strategy recommendations.

A lookup table, not a model. The mapping from (trend, condition) to
"box for slicks" is a dozen cells of domain knowledge that a race engineer can
read, audit and argue with - training something here would add opacity and
nothing else.

Urgency is a separate axis from the message so the UI can colour-code without
string-matching on prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .labels import CLASS_DAMP, CLASS_DRY, CLASS_DRYING, CLASS_WET
from .trend import (
    TREND_DRYING,
    TREND_INSUFFICIENT,
    TREND_STABLE,
    TREND_WETTING,
    TrendResult,
)

__all__ = ["Recommendation", "recommend", "ACTION_HOLD", "ACTION_MONITOR", "ACTION_PREPARE", "ACTION_ACT"]

ACTION_HOLD = "HOLD"
ACTION_MONITOR = "MONITOR"
ACTION_PREPARE = "PREPARE"
ACTION_ACT = "ACT"

#: Urgency 0-3, used by the frontend for colour and emphasis.
_URGENCY = {ACTION_HOLD: 0, ACTION_MONITOR: 1, ACTION_PREPARE: 2, ACTION_ACT: 3}


@dataclass(frozen=True)
class Recommendation:
    message: str
    action: str
    urgency: int
    tire: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "action": self.action,
            "urgency": self.urgency,
            "tire": self.tire,
        }


# (trend, current_class) -> (message, action, recommended tire)
_RULES: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    # --- Improving conditions ---
    (TREND_DRYING, CLASS_WET): (
        "Track drying - tire change window approaching. Ready intermediates.",
        ACTION_PREPARE,
        "Intermediate",
    ),
    (TREND_DRYING, CLASS_DRYING): (
        "Dry line established and widening - slick window opening. Ready slicks.",
        ACTION_PREPARE,
        "Slick (standby)",
    ),
    (TREND_DRYING, CLASS_DAMP): (
        "Track drying fast - switch to slicks now, before the undercut.",
        ACTION_ACT,
        "Slick",
    ),
    (TREND_DRYING, CLASS_DRY): (
        "Track dry - box for slicks this lap.",
        ACTION_ACT,
        "Slick",
    ),
    # --- Deteriorating ---
    (TREND_WETTING, CLASS_DRY): (
        "Rain arriving - surface darkening. Prepare intermediates.",
        ACTION_PREPARE,
        "Intermediate (standby)",
    ),
    (TREND_WETTING, CLASS_DAMP): (
        "Track wetting - switch to intermediates now.",
        ACTION_ACT,
        "Intermediate",
    ),
    (TREND_WETTING, CLASS_DRYING): (
        "Dry line closing up - conditions deteriorating. Intermediates now.",
        ACTION_ACT,
        "Intermediate",
    ),
    (TREND_WETTING, CLASS_WET): (
        "Standing water building - full wets, consider staying out of traffic.",
        ACTION_ACT,
        "Full Wet",
    ),
    # --- Stable ---
    (TREND_STABLE, CLASS_DRY): (
        "Track dry - no action needed.",
        ACTION_HOLD,
        "Slick",
    ),
    (TREND_STABLE, CLASS_DAMP): (
        "Track damp and stable - intermediates optimal, hold.",
        ACTION_HOLD,
        "Intermediate",
    ),
    (TREND_STABLE, CLASS_DRYING): (
        "Dry line forming but not yet established - monitor closely.",
        ACTION_MONITOR,
        "Intermediate",
    ),
    (TREND_STABLE, CLASS_WET): (
        "Track wet, hold current strategy - full wets.",
        ACTION_HOLD,
        "Full Wet",
    ),
}

_INSUFFICIENT = (
    "Collecting frames - trend not yet available.",
    ACTION_MONITOR,
    "Unknown",
)

#: Escalation ladder, used when a forecast contradicts what the camera sees.
_LADDER = [ACTION_HOLD, ACTION_MONITOR, ACTION_PREPARE, ACTION_ACT]

_RAIN_WORDS = ("rain", "shower", "storm", "drizzle", "downpour", "thunder", "wet")
_CLEAR_WORDS = ("clear", "dry", "sun", "sunny", "no rain", "clearing")


def forecast_signal(hint: Optional[str]) -> Optional[str]:
    """Read a free-text weather note as "rain", "clear", or nothing.

    Deliberately a keyword match rather than a forecast API call. A network
    dependency in the request path is a failure mode, and the note is an
    operator typing what they have been told, not a structured feed.
    """
    if not hint:
        return None
    text = hint.lower()
    # "no rain" contains "rain", so the clear signal is checked first.
    if any(word in text for word in _CLEAR_WORDS):
        return "clear"
    if any(word in text for word in _RAIN_WORDS):
        return "rain"
    return None


def _escalate(action: str) -> str:
    index = _LADDER.index(action)
    return _LADDER[min(index + 1, len(_LADDER) - 1)]


def recommend(result: TrendResult, weather_hint: Optional[str] = None) -> Recommendation:
    """Map a trend verdict to a pit-wall message.

    `weather_hint` is the brief's optional weather input: a free-text note such
    as "rain forecast in 10 min". It never overrides what the camera sees - it
    only appends context, because a forecast that disagrees with the track is
    the situation this tool is for.
    """
    if result.trend == TREND_INSUFFICIENT:
        message, action, tire = _INSUFFICIENT
    else:
        message, action, tire = _RULES[(result.trend, result.current_class)]

        # A fast-moving trend is worth flagging even when the rule says PREPARE.
        if result.trend_strength >= 0.75 and action == ACTION_PREPARE:
            message = f"{message} Conditions changing rapidly."

    # The camera stays authoritative: a forecast can raise the alert level by one
    # step but never changes the condition or the tyre. A note that agrees with
    # the track adds nothing, so only a genuine disagreement moves the needle -
    # rain expected while the surface is still dry is the case this tool exists
    # to catch early. Escalation stops short of ACT, because committing a car to
    # the pit lane on the strength of a typed note would be wrong.
    signal = forecast_signal(weather_hint)
    conflict = (
        signal == "rain"
        and result.trend != TREND_WETTING
        and result.current_class in (CLASS_DRY, CLASS_DAMP)
    )
    if conflict and action != ACTION_ACT:
        action = _escalate(action)
        if action == ACTION_ACT:  # never let a note alone demand a stop
            action = ACTION_PREPARE
        message = f"{message} Rain forecast while the track is not yet wetting - prepare early."

    if weather_hint:
        message = f"{message} (Weather note: {weather_hint})"

    return Recommendation(
        message=message,
        action=action,
        urgency=_URGENCY[action],
        tire=tire,
    )
