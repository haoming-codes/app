"""Configuration dataclasses for phonetic distance calculations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

SegmentMetricName = Literal["panphon", "phonetic_edit", "aline", "clts"]
ToneStrategyName = Literal["weighted", "none"]


@dataclass(slots=True)
class ToneConfusionMatrix:
    """Confusion matrix for Mandarin tones used during substitution costs."""

    substitution_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    insertion_cost: float = 1.0
    deletion_cost: float = 1.0

    def cost(self, tone_a: int, tone_b: int) -> float:
        """Return substitution cost between two tones."""
        if tone_a == tone_b:
            return 0.0
        return self.substitution_costs.get((tone_a, tone_b)) or self.substitution_costs.get(
            (tone_b, tone_a),
            1.0,
        )


DEFAULT_TONE_CONFUSION = ToneConfusionMatrix(
    substitution_costs={
        (1, 2): 0.4,
        (1, 3): 0.6,
        (1, 4): 0.8,
        (2, 3): 0.5,
        (2, 4): 0.7,
        (3, 4): 0.6,
        (1, 5): 0.5,
        (2, 5): 0.4,
        (3, 5): 0.5,
        (4, 5): 0.9,
    },
    insertion_cost=0.8,
    deletion_cost=0.8,
)


@dataclass(slots=True)
class ToneDistanceConfig:
    """Configuration for tone distance handling."""

    strategy: ToneStrategyName = "weighted"
    confusion: ToneConfusionMatrix = field(default_factory=lambda: deepcopy(DEFAULT_TONE_CONFUSION))
    normalize: bool = True


@dataclass(slots=True)
class DistanceConfig:
    """Top-level configuration for the phonetic distance calculator."""

    segment_metric: SegmentMetricName = "panphon"
    tone_config: ToneDistanceConfig = field(
        default_factory=lambda: ToneDistanceConfig(confusion=deepcopy(DEFAULT_TONE_CONFUSION))
    )
    tradeoff_lambda: float = 0.7
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tradeoff_lambda <= 1.0:
            raise ValueError("tradeoff_lambda must be in [0, 1]")


DEFAULT_DISTANCE_CONFIG = DistanceConfig()
