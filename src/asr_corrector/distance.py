"""Distance calculation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance

from .config import CorrectionConfig, MetricConfig
from .phonetics import CLTSSpace, ipa_for_text, tone_distance


@dataclass
class DistanceBreakdown:
    """Breakdown of the composite distance."""

    segmental: float
    tone: float
    total: float
    per_metric: Dict[str, float]


class DistanceCalculator:
    """Calculate distances according to a configuration."""

    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config
        self._panphon = PanphonDistance()
        self._phonetic_edit = PhoneticEditDistance()
        self._aline = ALINE()
        self._clts: Optional[CLTSSpace] = None

    def _ensure_clts(self) -> CLTSSpace:
        if self._clts is None:
            self._clts = CLTSSpace()
        return self._clts

    def _metric_distance(self, metric: MetricConfig, ipa_a: str, ipa_b: str, detoned_a: str, detoned_b: str) -> float:
        name = metric.name.lower()
        if name == "panphon":
            return self._panphon.weighted_feature_edit_distance(detoned_a, detoned_b)
        if name == "phonetic_edit":
            return self._phonetic_edit.dist(detoned_a, detoned_b)
        if name == "aline":
            return self._aline.dist(detoned_a, detoned_b)
        if name == "clts":
            return self._ensure_clts().distance(ipa_a, ipa_b)
        raise ValueError(f"Unknown metric: {metric.name}")

    def distance(self, text_a: str, text_b: str) -> DistanceBreakdown:
        rep_a = ipa_for_text(text_a)
        rep_b = ipa_for_text(text_b)
        per_metric: Dict[str, float] = {}
        weight_total = 0.0
        segment_total = 0.0
        for metric in self.config.metrics:
            dist = self._metric_distance(metric, rep_a.ipa, rep_b.ipa, rep_a.detoned_ipa, rep_b.detoned_ipa)
            per_metric[metric.name] = dist
            segment_total += metric.weight * dist
            weight_total += metric.weight
        segment_score = segment_total / weight_total if weight_total else 0.0

        tone_score = 0.0
        if self.config.tone and (rep_a.tones or rep_b.tones):
            tone_conf = self.config.tone.confusion_costs
            tone_score = tone_distance(
                rep_a.tones,
                rep_b.tones,
                default_cost=self.config.tone.default_cost,
                confusion_costs=tone_conf,
            )
            tone_score *= self.config.tone.weight

        total = self.config.segment_lambda * segment_score + self.config.tone_lambda * tone_score
        return DistanceBreakdown(
            segmental=segment_score,
            tone=tone_score,
            total=total,
            per_metric=per_metric,
        )

    def distance_value(self, text_a: str, text_b: str) -> float:
        """Convenience method returning only the total distance."""
        return self.distance(text_a, text_b).total
