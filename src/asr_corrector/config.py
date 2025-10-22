"""Configuration objects for phonetic distance calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


SegmentMetric = Literal["phonetic_edit", "aline"]
FeatureMetric = Literal["cosine", "euclidean"]


@dataclass
class DistanceConfig:
    """Configuration for combining phonetic, feature, and suprasegmental distances."""

    segment_metric: SegmentMetric = "phonetic_edit"
    feature_metric: FeatureMetric = "cosine"
    tone_weight: float = 1.0
    stress_weight: float = 1.0
    segment_weight: float = 1.0
    feature_weight: float = 1.0
    tone_penalty: float = 1.0
    stress_penalty: float = 1.0
    threshold: float = 0.5
    max_window: int = 6


@dataclass
class MatchingConfig:
    """Configuration for matching ASR output against a knowledge base."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    window_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    decision_threshold: float = 0.5
    require_tone_match: bool = False
    require_stress_match: bool = False
    allow_partial_windows: bool = True
    language_hint: Optional[str] = None
