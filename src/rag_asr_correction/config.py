"""Configuration objects for the RAG-inspired ASR corrector."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Tuple


class SegmentalMetric(str, Enum):
    """Supported segmental distance metrics."""

    PANPHON_WEIGHTED = "panphon_weighted"
    ABYDOS_PHONETIC_EDIT = "abydos_phonetic_edit"
    ABYDOS_ALINE = "abydos_aline"


@dataclass(frozen=True)
class SegmentalMetricConfig:
    """Configuration describing a segmental distance metric and its weight."""

    metric: SegmentalMetric
    weight: float = 1.0


@dataclass(frozen=True)
class ToneDistanceConfig:
    """Configuration for tone distance calculation."""

    weight: float = 1.0
    substitution_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    insertion_cost: float = 1.0
    deletion_cost: float = 1.0

    def cost(self, a: int, b: int) -> float:
        """Return the substitution cost between tone ``a`` and ``b``."""

        if a == b:
            return 0.0
        if (a, b) in self.substitution_costs:
            return self.substitution_costs[(a, b)]
        if (b, a) in self.substitution_costs:
            return self.substitution_costs[(b, a)]
        return 1.0


@dataclass(frozen=True)
class CorrectionConfig:
    """Configuration for the correction engine."""

    segmental_metrics: Iterable[SegmentalMetricConfig]
    tone_config: ToneDistanceConfig = ToneDistanceConfig()
    threshold: float = 2.0
    tradeoff_lambda: float = 1.0
    max_window_size: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "segmental_metrics", tuple(self.segmental_metrics))

    def metric_weights(self) -> List[Tuple[SegmentalMetric, float]]:
        """Return metric-weight pairs as a list."""

        return [(metric.metric, metric.weight) for metric in self.segmental_metrics]
