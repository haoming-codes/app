"""Configuration dataclasses for the phonetic RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass
class DistanceAggregationConfig:
    """Weights and knobs controlling distance aggregation."""

    segment_metric_weights: Dict[str, float] = field(
        default_factory=lambda: {"phonetic_edit_distance": 0.5, "aline": 0.5}
    )
    feature_weight: float = 1.0
    tone_weight: float = 1.0
    stress_weight: float = 1.0
    lambda_segment: float = 0.6
    lambda_tone: float = 0.25
    lambda_stress: float = 0.15
    dtw_local_metric: str = "cosine"
    treat_all_caps_as_acronyms: bool = True
    max_length_delta: int = 1

    def normalized_segment_weights(self) -> Dict[str, float]:
        total = sum(self.segment_metric_weights.values())
        if total <= 0:
            raise ValueError("segment_metric_weights must sum to a positive value")
        return {name: weight / total for name, weight in self.segment_metric_weights.items()}


@dataclass
class PhoneticPipelineConfig:
    """High-level knobs for the corrector."""

    aggregation: DistanceAggregationConfig = field(default_factory=DistanceAggregationConfig)
    window_threshold: float = 0.45
    max_candidates: int = 3
    languages: Sequence[str] = ("en", "cmn")
    tone_penalty_per_mismatch: float = 1.0
    stress_penalty_per_mismatch: float = 1.0

    def validate(self) -> None:
        if not 0 <= self.aggregation.lambda_segment <= 1:
            raise ValueError("lambda_segment must be between 0 and 1")
        if not 0 <= self.aggregation.lambda_tone <= 1:
            raise ValueError("lambda_tone must be between 0 and 1")
        if not 0 <= self.aggregation.lambda_stress <= 1:
            raise ValueError("lambda_stress must be between 0 and 1")
        total = (
            self.aggregation.lambda_segment
            + self.aggregation.lambda_tone
            + self.aggregation.lambda_stress
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Lambdas must sum to 1.0 for meaningful aggregation")
