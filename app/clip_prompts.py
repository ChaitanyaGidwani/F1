"""Prompt bank for the CLIP zero-shot path.

Kept in one place because two very different consumers depend on it and they
must not drift apart: the dataset triage step (data_pipeline/presort_clip.py)
and the runtime fallback classifier (app/inference.py).

Prompts describe the *surface*, never the weather. "a photo of a rainy day"
matches an overcast sky above a perfectly dry track, which is the single most
common zero-shot failure on this task.
"""

from __future__ import annotations

from typing import Dict, List

from .labels import CLASS_DAMP, CLASS_DRY, CLASS_DRYING, CLASS_WET

REJECT = "Reject"

CLASS_PROMPTS: Dict[str, List[str]] = {
    CLASS_DRY: [
        "a photo of a dry racetrack with pale grey asphalt",
        "a race car on a completely dry track surface",
        "dry asphalt racing surface in sunshine",
    ],
    CLASS_DAMP: [
        "a photo of a damp racetrack, dark asphalt with no standing water",
        "a race track surface that is wet but has no puddles or spray",
        "slightly damp dark grey asphalt after light rain",
    ],
    CLASS_DRYING: [
        "a racetrack with a dry racing line on otherwise wet asphalt",
        "a drying race track with dry strips and wet patches",
        "a partially dry race track surface with a visible dry line",
    ],
    CLASS_WET: [
        "a soaking wet racetrack with standing water and heavy spray",
        "a race car throwing up a huge plume of water spray in the rain",
        "a flooded race track with puddles and reflections in heavy rain",
    ],
}

#: Only used by dataset triage - the runtime classifier must always return one
#: of the four real classes.
REJECT_PROMPTS: List[str] = [
    "a portrait photograph of a person",
    "a racing car parked in a garage or pit box",
    "a crowd of spectators in a grandstand",
    "a close-up of a car wheel or engine part",
    "a trophy presentation on a podium",
    "a logo, sign, poster or text document",
    "an aerial map or circuit layout diagram",
    "the interior of a building",
]
