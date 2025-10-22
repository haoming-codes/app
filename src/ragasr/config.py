"""Configuration objects for phonetic distance calculation and correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


EditMetric = Literal["phonetic", "aline"]
FeatureDistance = Literal["cosine", "manhattan", "euclidean"]


@dataclass(slots=True)
class DistanceConfig:
    """Configuration for phonetic distance calculation."""

    threshold: float = 0.2
    segment_metrics: Sequence[EditMetric] = ("phonetic", "aline")
    feature_metric: FeatureDistance = "cosine"
    use_feature_dtw: bool = True
    segment_weight: float = 0.6
    tone_weight: float = 0.2
    stress_weight: float = 0.2
    tone_penalty: float = 1.0
    stress_penalty: float = 1.0

    def __post_init__(self) -> None:
        total = self.segment_weight + self.tone_weight + self.stress_weight
        if total <= 0:
            raise ValueError("weights must sum to a positive value")


@dataclass(slots=True)
class CorrectionConfig:
    """Configuration for the ASR corrector."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    window_radius: int = 2
    allow_overwrite: bool = False
