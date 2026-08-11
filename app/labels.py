"""Class vocabulary and the wetness axis the trend logic runs on.

The four frame-level classes are ordered along a single physical quantity: how
much standing water is on the racing surface. Putting them on a numeric axis is
what makes "is the track getting better or worse?" answerable with a slope
instead of a pile of if-statements.

    Dry     0.0   uniform light-grey asphalt, no sheen
    Damp    1.0   uniformly darkened asphalt, no standing water, no spray
    Drying  1.5   heterogeneous: a dry racing line cutting through a wet surface
    Wet     2.0   standing water, heavy reflections, rooster tails of spray

`Drying` sits at 1.5 rather than between Dry and Damp. A track with a
dry line forming still holds a lot of water off-line - it is physically wetter
than a uniformly damp track. Its value as a *signal* is that it is strong
visual evidence conditions are improving, which the trend layer picks up
separately from the raw wetness value.
"""

from __future__ import annotations

from typing import Dict, List

CLASS_DRY = "Dry"
CLASS_DAMP = "Damp"
CLASS_DRYING = "Drying"
CLASS_WET = "Wet"

#: Canonical class order, ascending wetness. Model label ids follow this order.
CLASSES: List[str] = [CLASS_DRY, CLASS_DAMP, CLASS_DRYING, CLASS_WET]

#: Position of each class on the wetness axis.
WETNESS: Dict[str, float] = {
    CLASS_DRY: 0.0,
    CLASS_DAMP: 1.0,
    CLASS_DRYING: 1.5,
    CLASS_WET: 2.0,
}

WETNESS_MIN = 0.0
WETNESS_MAX = 2.0

#: Colour hints shared by the web UI and the Gradio Space so the two demos agree.
CLASS_COLORS: Dict[str, str] = {
    CLASS_DRY: "#22c55e",
    CLASS_DAMP: "#eab308",
    CLASS_DRYING: "#38bdf8",
    CLASS_WET: "#ef4444",
}

_LOOKUP = {c.lower(): c for c in CLASSES}
# Tolerate the vocabularies of the upstream road-surface datasets we borrow from.
_LOOKUP.update(
    {
        "dry_asphalt": CLASS_DRY,
        "dry-asphalt": CLASS_DRY,
        "wet_asphalt": CLASS_DAMP,
        "wet-asphalt": CLASS_DAMP,
        "water_asphalt": CLASS_WET,
        "water-asphalt": CLASS_WET,
        "water": CLASS_WET,
        "damp_asphalt": CLASS_DAMP,
        "drying_asphalt": CLASS_DRYING,
    }
)


def canonical(label: str) -> str:
    """Normalise a label from a model, folder name or dataset into a canon class.

    Raises ValueError on an unknown label rather than silently guessing - a
    mislabelled class would quietly corrupt the wetness axis.
    """
    key = str(label).strip().lower().replace(" ", "_")
    if key in _LOOKUP:
        return _LOOKUP[key]
    raise ValueError(f"Unknown track-condition label {label!r}; expected one of {CLASSES}")


def wetness_of(label: str) -> float:
    """Wetness-axis position for a class label."""
    return WETNESS[canonical(label)]
