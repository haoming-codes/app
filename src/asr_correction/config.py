"""Configuration structures for phonetic distance computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SegmentalMetricConfig:
    """Configuration for an individual segmental distance metric."""

    name: str
    weight: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToneDistanceConfig:
    """Configuration for tone distance computation."""

    weight: float = 1.0
    confusion_penalties: Mapping[tuple[str, str], float] = field(default_factory=dict)
    insertion_penalty: float = 1.0
    deletion_penalty: float = 1.0

    def penalty(self, source: str, target: str) -> float:
        """Return substitution penalty for a tone pair."""

        if source == target:
            return 0.0
        penalty = self.confusion_penalties.get((source, target))
        if penalty is not None:
            return penalty
        penalty = self.confusion_penalties.get((target, source))
        if penalty is not None:
            return penalty
        return 1.0


@dataclass(frozen=True)
class DistanceComputationConfig:
    """Configuration for computing the overall phonetic distance."""

    segmental_metrics: Iterable[SegmentalMetricConfig] = field(default_factory=list)
    tone: Optional[ToneDistanceConfig] = None
    tradeoff_lambda: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.tradeoff_lambda <= 1.0:
            raise ValueError("tradeoff_lambda must be in the [0, 1] range")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")


DEFAULT_SEGMENTAL_METRICS = (
    SegmentalMetricConfig("panphon"),
    SegmentalMetricConfig("phonetic_edit_distance"),
    SegmentalMetricConfig("aline"),
)
"""A reasonable default set of segmental metrics."""


DEFAULT_CONFIG = DistanceComputationConfig(
    segmental_metrics=DEFAULT_SEGMENTAL_METRICS,
    tone=ToneDistanceConfig(),
    tradeoff_lambda=0.75,
    threshold=0.45,
)
"""Default configuration combining segmental and tone distances."""
