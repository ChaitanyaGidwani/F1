"""Turn a stream of per-frame predictions into a track-condition *trend*.

A single frame can't express "drying"; it's a sequence signal. This keeps the
last N classifier outputs and works out whether conditions are improving or
deteriorating.

Two signals are combined:

1. Wetness slope. Each frame's probability vector is projected onto the wetness
   axis in labels.py to give a scalar, then a least-squares fit over the window
   gives a slope. Negative means drying.
2. Dominant-class transition (Wet -> Damp, Damp -> Dry), found by comparing the
   first half of the buffer against the second.

A trend is only reported when the slope clears a threshold and the net change
between the two halves agrees with it in sign and size. Requiring both stops a
classifier that flickers between adjacent classes from looking like weather.

No torch or FastAPI imports here, so this can be tested against synthetic
sequences without loading a model. See tests/test_trend.py.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

from .labels import CLASSES, WETNESS, canonical

__all__ = [
    "FramePrediction",
    "TrendConfig",
    "TrendResult",
    "TrackConditionTracker",
    "analyze",
    "TREND_DRYING",
    "TREND_WETTING",
    "TREND_STABLE",
    "TREND_INSUFFICIENT",
]

TREND_DRYING = "DRYING"
TREND_WETTING = "WETTING"
TREND_STABLE = "STABLE"
TREND_INSUFFICIENT = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class FramePrediction:
    """One frame's classifier output."""

    label: str
    confidence: float
    probs: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def from_probs(probs: Mapping[str, float], timestamp: Optional[float] = None) -> "FramePrediction":
        """Build from a full probability distribution (the preferred path)."""
        clean = {c: 0.0 for c in CLASSES}
        for raw, p in probs.items():
            clean[canonical(raw)] += float(p)
        total = sum(clean.values())
        if total <= 0:
            raise ValueError("Probability vector sums to zero")
        clean = {c: v / total for c, v in clean.items()}
        top = max(clean, key=lambda c: clean[c])
        return FramePrediction(
            label=top,
            confidence=clean[top],
            probs=clean,
            timestamp=time.time() if timestamp is None else timestamp,
        )

    @staticmethod
    def from_label(
        label: str, confidence: float = 1.0, timestamp: Optional[float] = None
    ) -> "FramePrediction":
        """Build from a top-1 label only, spreading the remaining mass evenly.

        Useful for tests and for any upstream model that only exposes argmax.
        """
        top = canonical(label)
        confidence = min(max(float(confidence), 0.0), 1.0)
        rest = (1.0 - confidence) / max(len(CLASSES) - 1, 1)
        probs = {c: (confidence if c == top else rest) for c in CLASSES}
        return FramePrediction.from_probs(probs, timestamp=timestamp)

    @property
    def wetness(self) -> float:
        """Expected wetness under this frame's distribution."""
        return sum(self.probs[c] * WETNESS[c] for c in CLASSES)


@dataclass(frozen=True)
class TrendConfig:
    """Tunables for trend detection.

    Defaults are set for a ~1 fps trackside feed over a ~12 second window.
    """

    window: int = 12
    #: Below this many frames we refuse to call a trend at all.
    min_frames: int = 4
    #: EMA weight on the newest frame when smoothing the class distribution.
    ema_alpha: float = 0.4
    #: Minimum |slope| (wetness units per frame) to consider a trend real.
    slope_threshold: float = 0.025
    #: Minimum |mean(second half) - mean(first half)| to confirm the slope.
    min_net_delta: float = 0.15

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")


DEFAULT_CONFIG = TrendConfig()


@dataclass(frozen=True)
class TrendResult:
    """Everything the API and UI need to render one update."""

    current_class: str
    current_confidence: float
    instant_class: str
    instant_confidence: float
    trend: str
    trend_strength: float
    wetness: float
    slope: float
    transition: Optional[str]
    frames: int
    wetness_history: List[float]
    confidence_history: List[float]
    class_history: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_class": self.current_class,
            "current_confidence": round(self.current_confidence, 4),
            "instant_class": self.instant_class,
            "instant_confidence": round(self.instant_confidence, 4),
            "trend": self.trend,
            "trend_strength": round(self.trend_strength, 4),
            "wetness": round(self.wetness, 4),
            "slope": round(self.slope, 5),
            "transition": self.transition,
            "frames": self.frames,
            "wetness_history": [round(w, 4) for w in self.wetness_history],
            "confidence_history": [round(c, 4) for c in self.confidence_history],
            "class_history": list(self.class_history),
        }


def _mean_probs(frames: Sequence[FramePrediction]) -> Dict[str, float]:
    acc = {c: 0.0 for c in CLASSES}
    for f in frames:
        for c in CLASSES:
            acc[c] += f.probs.get(c, 0.0)
    n = max(len(frames), 1)
    return {c: v / n for c, v in acc.items()}


def _dominant(probs: Mapping[str, float]) -> str:
    return max(CLASSES, key=lambda c: probs.get(c, 0.0))


def _ema_probs(frames: Sequence[FramePrediction], alpha: float) -> Dict[str, float]:
    """Exponentially-weighted class distribution, newest frame weighted most."""
    smoothed = dict(frames[0].probs)
    for f in frames[1:]:
        for c in CLASSES:
            smoothed[c] = alpha * f.probs.get(c, 0.0) + (1.0 - alpha) * smoothed[c]
    total = sum(smoothed.values()) or 1.0
    return {c: v / total for c, v in smoothed.items()}


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _least_squares_slope(values: Sequence[float]) -> float:
    """Slope of a straight-line fit through `values` against frame index."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def analyze(
    frames: Sequence[FramePrediction], config: TrendConfig = DEFAULT_CONFIG
) -> TrendResult:
    """Reduce a sequence of frame predictions to a trend verdict.

    Pure function: same input always gives the same output, no model or network
    involved. Only the most recent `config.window` frames are considered.
    """
    window = list(frames)[-config.window :]

    if not window:
        return TrendResult(
            current_class=CLASSES[0],
            current_confidence=0.0,
            instant_class=CLASSES[0],
            instant_confidence=0.0,
            trend=TREND_INSUFFICIENT,
            trend_strength=0.0,
            wetness=0.0,
            slope=0.0,
            transition=None,
            frames=0,
            wetness_history=[],
            confidence_history=[],
            class_history=[],
        )

    wetness_history = [f.wetness for f in window]
    confidence_history = [f.confidence for f in window]
    class_history = [f.label for f in window]

    smoothed = _ema_probs(window, config.ema_alpha)
    current_class = _dominant(smoothed)
    current_confidence = smoothed[current_class]
    current_wetness = sum(smoothed[c] * WETNESS[c] for c in CLASSES)

    latest = window[-1]

    # Not enough history to say anything about direction yet.
    if len(window) < config.min_frames:
        return TrendResult(
            current_class=current_class,
            current_confidence=current_confidence,
            instant_class=latest.label,
            instant_confidence=latest.confidence,
            trend=TREND_INSUFFICIENT,
            trend_strength=0.0,
            wetness=current_wetness,
            slope=0.0,
            transition=None,
            frames=len(window),
            wetness_history=wetness_history,
            confidence_history=confidence_history,
            class_history=class_history,
        )

    slope = _least_squares_slope(wetness_history)

    half = len(window) // 2
    first_half, second_half = window[:half], window[half:]
    first_probs, second_probs = _mean_probs(first_half), _mean_probs(second_half)

    # Median wetness of each half, not the mean. A single
    # misclassified frame - a car throwing spray across the lens, a sun flare on
    # wet asphalt - moves a 6-frame mean by ~0.3 wetness units, which is enough
    # to clear min_net_delta and fake a weather change. The median ignores it,
    # so a trend has to be carried by the majority of the window to be reported.
    first_wet = _median([f.wetness for f in first_half])
    second_wet = _median([f.wetness for f in second_half])
    net_delta = second_wet - first_wet

    # Both signals must agree before a trend is declared. The slope says the
    # movement is sustained; the net delta says it is actually large enough to
    # matter and is robust to a noisy frame at an edge.
    trend = TREND_STABLE
    if (
        abs(slope) >= config.slope_threshold
        and abs(net_delta) >= config.min_net_delta
        and (slope < 0) == (net_delta < 0)
    ):
        trend = TREND_DRYING if slope < 0 else TREND_WETTING

    # The brief's explicit framing: name the dominant-class shift, e.g. "Wet->Damp".
    transition: Optional[str] = None
    first_dom, second_dom = _dominant(first_probs), _dominant(second_probs)
    if first_dom != second_dom and trend in (TREND_DRYING, TREND_WETTING):
        transition = f"{first_dom}->{second_dom}"

    strength = 0.0
    if trend in (TREND_DRYING, TREND_WETTING):
        strength = min(1.0, abs(slope) / (config.slope_threshold * 4.0))

    return TrendResult(
        current_class=current_class,
        current_confidence=current_confidence,
        instant_class=latest.label,
        instant_confidence=latest.confidence,
        trend=trend,
        trend_strength=strength,
        wetness=current_wetness,
        slope=slope,
        transition=transition,
        frames=len(window),
        wetness_history=wetness_history,
        confidence_history=confidence_history,
        class_history=class_history,
    )


class TrackConditionTracker:
    """Stateful rolling buffer around `analyze` - one per camera/session."""

    def __init__(self, config: TrendConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._buffer: Deque[FramePrediction] = deque(maxlen=config.window)

    def update(self, prediction: FramePrediction) -> TrendResult:
        """Add a frame and return the trend over the buffer including it."""
        self._buffer.append(prediction)
        return self.analyze()

    def analyze(self) -> TrendResult:
        return analyze(list(self._buffer), self.config)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def frames(self) -> List[FramePrediction]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
