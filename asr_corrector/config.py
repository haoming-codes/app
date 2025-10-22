from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class SegmentMetricConfig:
    """Configuration for a segment-level distance metric."""

    name: str
    weight: float = 1.0
    factory: Callable[[], object] | None = None


@dataclass
class SegmentDistanceConfig:
    """Configuration controlling segment similarity computations."""

    metrics: List[SegmentMetricConfig] = field(
        default_factory=lambda: [
            SegmentMetricConfig(name="phonetic_edit_distance", weight=1.0),
            SegmentMetricConfig(name="aline", weight=1.0),
        ]
    )
    feature_weight: float = 1.0
    metrics_weight: float = 1.0
    feature_distance: str = "l1"


@dataclass
class DistanceWeights:
    """Weights applied to the combined distance components."""

    segment: float = 0.6
    tone: float = 0.2
    stress: float = 0.2


@dataclass
class DistanceConfig:
    """Full configuration for distance computation and correction."""

    segment: SegmentDistanceConfig = field(default_factory=SegmentDistanceConfig)
    weights: DistanceWeights = field(default_factory=DistanceWeights)
    tone_alignment: str = "levenshtein"
    stress_alignment: str = "levenshtein"
    correction_threshold: float = 0.4
    window_radius: int = 1

