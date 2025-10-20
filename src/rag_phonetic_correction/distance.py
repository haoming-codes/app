"""Distance utilities that combine multiple phonetic metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance

from .config import PhoneticDistanceConfig


@dataclass
class SegmentalScores:
    panphon: float = 0.0
    aline: float = 0.0
    phonetic_edit: float = 0.0

    def weighted_sum(self, config: PhoneticDistanceConfig) -> float:
        total = 0.0
        weight_sum = 0.0
        for name, weight in config.metric_weights().items():
            if weight <= 0:
                continue
            total += weight * getattr(self, name)
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0


class PhoneticDistanceCalculator:
    """Combine segmental distances with tone penalties."""

    def __init__(self, config: PhoneticDistanceConfig) -> None:
        self._config = config
        self._panphon = Distance()
        self._aline = ALINE()
        self._phonetic_edit = PhoneticEditDistance()

    def segmental_distance(self, ipa_a: str, ipa_b: str) -> SegmentalScores:
        scores = SegmentalScores()
        if self._config.panphon_weight > 0:
            scores.panphon = self._panphon.weighted_feature_edit_distance(ipa_a, ipa_b)
        if self._config.aline_weight > 0:
            scores.aline = self._aline.distance(ipa_a, ipa_b)
        if self._config.phonetic_edit_weight > 0:
            scores.phonetic_edit = self._phonetic_edit.distance(ipa_a, ipa_b)
        return scores

    def tone_distance(self, tones_a: Sequence[int], tones_b: Sequence[int]) -> float:
        if self._config.tone_weight <= 0:
            return 0.0
        gap = self._config.tone_gap_penalty
        m = len(tones_a)
        n = len(tones_b)
        if m == 0 and n == 0:
            return 0.0
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i * gap
        for j in range(1, n + 1):
            dp[0][j] = j * gap
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                substitution = dp[i - 1][j - 1] + self._tone_penalty(tones_a[i - 1], tones_b[j - 1])
                deletion = dp[i - 1][j] + gap
                insertion = dp[i][j - 1] + gap
                dp[i][j] = min(substitution, deletion, insertion)
        normalization = max(m, n, 1) ** self._config.normalization_exponent
        return dp[m][n] / normalization

    def total_distance(self, ipa_a: str, ipa_b: str, tones_a: Sequence[int], tones_b: Sequence[int]) -> float:
        scores = self.segmental_distance(ipa_a, ipa_b)
        segmental = scores.weighted_sum(self._config)
        tone = self.tone_distance(tones_a, tones_b)
        return segmental + self._config.tone_weight * tone

    def _tone_penalty(self, tone_a: int, tone_b: int) -> float:
        if tone_a == tone_b:
            return 0.0
        confusion = self._config.tone_confusion
        if confusion:
            value = confusion.get((tone_a, tone_b))
            if value is None:
                value = confusion.get((tone_b, tone_a))
            if value is not None:
                return value
        return self._config.tone_default_penalty
