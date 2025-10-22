"""Configuration objects for phonetic distance scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

SegmentMetric = Literal["aline", "phonetic_edit"]
FeatureMetric = Literal["l1", "l2"]


@dataclass(slots=True)
class PhoneticScoringConfig:
    """Hyper-parameters for combining phonetic distance components."""

    segment_metric: SegmentMetric = "aline"
    segment_metric_kwargs: Mapping[str, float] | None = None
    feature_metric: FeatureMetric = "l1"
    segment_weight: float = 0.6
    dtw_weight: float = 0.3
    tone_weight: float = 0.05
    stress_weight: float = 0.05
    tone_neutral: int = 5
    stress_primary: int = 1
    stress_secondary: int = 2

    def normalized_weights(self) -> tuple[float, float, float, float]:
        """Return the normalized weights for each component."""

        weights = [
            max(self.segment_weight, 0.0),
            max(self.dtw_weight, 0.0),
            max(self.tone_weight, 0.0),
            max(self.stress_weight, 0.0),
        ]
        total = sum(weights)
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0
        return tuple(weight / total for weight in weights)
