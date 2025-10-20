"""Configuration objects for the ASR correction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence


class SegmentalMetric(str, Enum):
    """Available segmental distance metrics."""

    PANPHON = "panphon"
    PHONETIC_EDIT = "phonetic_edit"
    ALINE = "aline"
    CLTS = "clts"


class FeatureDistance(str, Enum):
    """Distance functions for CLTS feature vectors."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@dataclass
class SegmentalMetricConfig:
    """Configuration for a single segmental metric."""

    metric: SegmentalMetric
    weight: float = 1.0
    feature_distance: FeatureDistance = FeatureDistance.COSINE


@dataclass
class ToneConfusionMatrix:
    """Stores a confusion-weighted table for Mandarin tones."""

    matrix: Dict[tuple[int, int], float] = field(default_factory=dict)
    default_cost: float = 1.0

    def cost(self, source: int, target: int) -> float:
        """Return the substitution cost between two tone numbers."""

        if source == target:
            return 0.0
        return self.matrix.get((source, target)) or self.matrix.get((target, source)) or self.default_cost


@dataclass
class DistanceConfig:
    """Configuration used to compute phonetic distances."""

    segmental_metrics: Sequence[SegmentalMetricConfig] = field(default_factory=list)
    tone_weight: float = 0.0
    tone_confusion: ToneConfusionMatrix = field(default_factory=ToneConfusionMatrix)
    segmental_scale: float = 1.0

    @classmethod
    def default_for_language(cls, language: str) -> "DistanceConfig":
        """Return a sensible default configuration for a given language code."""

        if language == "zh":
            return cls(
                segmental_metrics=[SegmentalMetricConfig(SegmentalMetric.PANPHON, weight=1.0)],
                tone_weight=0.5,
                tone_confusion=default_tone_confusions(),
            )
        return cls(segmental_metrics=[SegmentalMetricConfig(SegmentalMetric.PHONETIC_EDIT, weight=1.0)])


@dataclass
class MatcherConfig:
    """Configuration for sliding-window matching."""

    window_sizes: Sequence[int] = (1, 2, 3, 4)
    distance_threshold: float = 1.0
    tradeoff_lambda: float = 1.0
    segmental_language_map: Dict[str, str] = field(default_factory=lambda: {"zh": "cmn-Hans", "en": "eng-Latn"})
    max_candidates: int = 5

    def __post_init__(self) -> None:
        if not self.window_sizes:
            raise ValueError("window_sizes cannot be empty")


@dataclass
class KnowledgeBaseEntry:
    """Single entry in the knowledge base."""

    surface: str
    language: str
    metadata: Optional[dict] = None


def default_tone_confusions() -> ToneConfusionMatrix:
    """Return a heuristic confusion table for Mandarin tones."""

    table: Dict[tuple[int, int], float] = {}
    # Typical confusions: 2<->3 and 3<->4 are more likely than 1<->4 etc.
    for a, b, cost in [
        (2, 3, 0.3),
        (3, 4, 0.4),
        (2, 4, 0.6),
        (1, 2, 0.5),
        (1, 4, 0.7),
        (3, 1, 0.6),
    ]:
        table[(a, b)] = cost
    return ToneConfusionMatrix(matrix=table, default_cost=1.0)
