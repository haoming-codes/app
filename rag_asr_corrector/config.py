"""Configuration objects for phonetic correction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class DistanceConfig:
    """Hyper-parameters controlling distance computation."""

    segment_metrics: List[str] = field(
        default_factory=lambda: ["phonetic_edit_distance", "aline"]
    )
    feature_metric: str = "cosine"
    segment_weight: float = 0.6
    feature_weight: float = 0.3
    tone_weight: float = 0.1
    stress_weight: float = 0.0

    def validate(self) -> None:
        if not self.segment_metrics:
            raise ValueError("segment_metrics must contain at least one metric")
        allowed = {"phonetic_edit_distance", "aline"}
        for metric in self.segment_metrics:
            if metric not in allowed:
                raise ValueError(f"Unsupported segment metric: {metric}")
        if self.feature_metric not in {"cosine", "euclidean"}:
            raise ValueError("feature_metric must be 'cosine' or 'euclidean'")
        total = self.segment_weight + self.feature_weight + self.tone_weight + self.stress_weight
        if total <= 0:
            raise ValueError("At least one weight must be positive")


@dataclass
class CorrectionConfig:
    """Configuration for correction suggestions."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    threshold: float = 0.35
    window_tolerance: int = 1

    def validate(self) -> None:
        self.distance.validate()
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.window_tolerance < 0:
            raise ValueError("window_tolerance must be non-negative")


def normalize_weights(weights: Iterable[float]) -> List[float]:
    weights = list(weights)
    total = sum(weights)
    if total == 0:
        raise ValueError("Cannot normalise zero weights")
    return [w / total for w in weights]
