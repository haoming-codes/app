"""Configuration objects for phonetic distance calculations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DistanceConfig:
    """Hyper-parameters steering the composite phonetic distance.

    Attributes
    ----------
    segment_metrics:
        Mapping from segment-level metric names to weights. Supported metrics are
        ``"phonetic_edit_distance"``, ``"aline"``, and ``"dtw"``.
    lambda_segment:
        Weight of the segment distance in the final score.
    lambda_tone:
        Weight of the Mandarin tone penalty in the final score.
    lambda_stress:
        Weight of the English stress penalty in the final score.
    tone_penalty:
        Cost applied to a tone mismatch during alignment.
    stress_penalty:
        Cost applied to a stress mismatch during alignment.
    threshold:
        Maximum total distance that is considered a potential match when
        scanning ASR outputs against a knowledge base.
    window_expansion:
        How many tokens to expand candidate windows when searching for matches.
    """

    segment_metrics: Dict[str, float] = field(
        default_factory=lambda: {"phonetic_edit_distance": 0.4, "aline": 0.4, "dtw": 0.2}
    )
    lambda_segment: float = 0.5
    lambda_tone: float = 0.25
    lambda_stress: float = 0.25
    tone_penalty: float = 1.0
    stress_penalty: float = 1.0
    threshold: float = 0.45
    window_expansion: int = 1

    def normalized_weights(self) -> Dict[str, float]:
        """Return the segment metric weights normalized to a convex combination."""

        total = sum(self.segment_metrics.values())
        if total <= 0:
            return {name: 1.0 for name in self.segment_metrics}
        return {name: weight / total for name, weight in self.segment_metrics.items()}

    def normalized_lambdas(self) -> Dict[str, float]:
        """Return normalized component weights for the final combination."""

        total = self.lambda_segment + self.lambda_tone + self.lambda_stress
        if total <= 0:
            return {
                "segment": 1 / 3,
                "tone": 1 / 3,
                "stress": 1 / 3,
            }
        return {
            "segment": self.lambda_segment / total,
            "tone": self.lambda_tone / total,
            "stress": self.lambda_stress / total,
        }
