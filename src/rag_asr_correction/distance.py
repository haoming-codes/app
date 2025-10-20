"""Distance utilities for phonetic and tone-aware comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance

from .config import CorrectionConfig, SegmentalMetric
from .phonetics import PhoneticRepresentation, build_phonetic_representation


@dataclass
class DistanceBreakdown:
    """Stores detailed distance information."""

    segmental: Dict[SegmentalMetric, float]
    tone: float


class DistanceComputer:
    """Compute distances according to a :class:`CorrectionConfig`."""

    def __init__(self, config: CorrectionConfig):
        self.config = config
        self._panphon = PanphonDistance()
        self._abydos_phonetic = PhoneticEditDistance()
        self._abydos_aline = ALINE()

    def _segmental_distance(
        self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation, metric: SegmentalMetric
    ) -> float:
        if metric == SegmentalMetric.PANPHON_WEIGHTED:
            return self._panphon.weighted_feature_edit_distance(rep_a.segmental, rep_b.segmental)
        if metric == SegmentalMetric.ABYDOS_PHONETIC_EDIT:
            return self._abydos_phonetic.dist(rep_a.segmental, rep_b.segmental)
        if metric == SegmentalMetric.ABYDOS_ALINE:
            return self._abydos_aline.dist(rep_a.segmental, rep_b.segmental)
        raise ValueError(f"Unsupported metric: {metric}")

    def tone_distance(self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> float:
        seq_a = rep_a.tone_sequence
        seq_b = rep_b.tone_sequence
        if not seq_a and not seq_b:
            return 0.0

        tone_conf = self.config.tone_config
        len_a = len(seq_a)
        len_b = len(seq_b)
        dp = [[0.0 for _ in range(len_b + 1)] for _ in range(len_a + 1)]
        for i in range(1, len_a + 1):
            dp[i][0] = i * tone_conf.deletion_cost
        for j in range(1, len_b + 1):
            dp[0][j] = j * tone_conf.insertion_cost

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                deletion = dp[i - 1][j] + tone_conf.deletion_cost
                insertion = dp[i][j - 1] + tone_conf.insertion_cost
                substitution = dp[i - 1][j - 1] + tone_conf.cost(seq_a[i - 1], seq_b[j - 1])
                dp[i][j] = min(deletion, insertion, substitution)
        return dp[len_a][len_b]

    def segmental_distances(
        self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation
    ) -> Dict[SegmentalMetric, float]:
        distances: Dict[SegmentalMetric, float] = {}
        for metric, _ in self.config.metric_weights():
            distances[metric] = self._segmental_distance(rep_a, rep_b, metric)
        return distances

    def combined_distance(self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> DistanceBreakdown:
        segmental = self.segmental_distances(rep_a, rep_b)
        tone = self.tone_distance(rep_a, rep_b)
        return DistanceBreakdown(segmental=segmental, tone=tone)

    def score(self, breakdown: DistanceBreakdown) -> float:
        seg_total = 0.0
        for metric, weight in self.config.metric_weights():
            seg_total += weight * breakdown.segmental.get(metric, 0.0)
        tone_component = self.config.tone_config.weight * breakdown.tone
        return seg_total + self.config.tradeoff_lambda * tone_component

    def distance(self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> float:
        return self.score(self.combined_distance(rep_a, rep_b))

    def distance_between_strings(
        self, text_a: str, text_b: str, lang_a: str | None = None, lang_b: str | None = None
    ) -> float:
        rep_a = build_phonetic_representation(text_a, language=lang_a)
        rep_b = build_phonetic_representation(text_b, language=lang_b)
        return self.distance(rep_a, rep_b)
