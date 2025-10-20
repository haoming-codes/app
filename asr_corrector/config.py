from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional


class SegmentalMetric(Enum):
    PANPHON = "panphon"
    ABYDOS_PHONETIC = "abydos_phonetic"
    ABYDOS_ALINE = "abydos_aline"
    CLTS_VECTOR = "clts_vector"


class ToneMetric(Enum):
    NONE = "none"
    CONFUSION = "confusion"


@dataclass
class ToneConfig:
    metric: ToneMetric = ToneMetric.NONE
    confusion_costs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default_cost: float = 1.0

    def cost(self, from_tone: str, to_tone: str) -> float:
        if self.metric == ToneMetric.NONE:
            return 0.0
        if from_tone == to_tone:
            return 0.0
        return self.confusion_costs.get(from_tone, {}).get(to_tone, self.default_cost)


@dataclass
class SegmentalConfig:
    metric: SegmentalMetric = SegmentalMetric.PANPHON
    clts_vector_distance: str = "cosine"
    panphon_feature_weights: Optional[Iterable[float]] = None


@dataclass
class DistanceConfig:
    segmental: SegmentalConfig = field(default_factory=SegmentalConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    segmental_weight: float = 1.0
    tone_weight: float = 1.0
    threshold: float = 2.0


DEFAULT_TONE_CONFUSION = {
    "1": {"2": 0.7, "3": 0.9, "4": 1.0, "5": 0.5},
    "2": {"1": 0.7, "3": 0.6, "4": 0.9, "5": 0.6},
    "3": {"1": 0.9, "2": 0.6, "4": 0.7, "5": 0.8},
    "4": {"1": 1.0, "2": 0.9, "3": 0.7, "5": 0.9},
    "5": {"1": 0.5, "2": 0.6, "3": 0.8, "4": 0.9},
}


DEFAULT_DISTANCE_CONFIG = DistanceConfig(
    segmental=SegmentalConfig(metric=SegmentalMetric.PANPHON),
    tone=ToneConfig(metric=ToneMetric.CONFUSION, confusion_costs=DEFAULT_TONE_CONFUSION, default_cost=1.0),
    segmental_weight=1.0,
    tone_weight=0.5,
    threshold=2.5,
)
