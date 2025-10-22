"""Phonetic distance computation utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig, normalize_weights
from .ipa import IPAResult, MultilingualIPAConverter


@dataclass
class DistanceBreakdown:
    """Distances for different articulatory components."""

    segment_distance: float
    feature_distance: float
    tone_distance: float
    stress_distance: float
    total: float


class DistanceCalculator:
    """Compute aggregated phonetic distances between strings."""

    def __init__(self, config: DistanceConfig | None = None, converter: MultilingualIPAConverter | None = None) -> None:
        self.config = config or DistanceConfig()
        self.config.validate()
        self.converter = converter or MultilingualIPAConverter()
        self._ped = PhoneticEditDistance()
        self._aline = ALINE()

    def distance(self, a: str, b: str) -> DistanceBreakdown:
        ipa_a = self.converter.ipa(a)
        ipa_b = self.converter.ipa(b)
        return self.distance_from_results(ipa_a, ipa_b)

    def distance_from_results(self, ipa_a: IPAResult, ipa_b: IPAResult) -> DistanceBreakdown:
        segment_distance = self._segment_distance(ipa_a, ipa_b)
        feature_distance = self._feature_distance(ipa_a, ipa_b)
        tone_distance = self._tone_distance(ipa_a, ipa_b)
        stress_distance = self._stress_distance(ipa_a, ipa_b)

        weights = []
        components = []

        if self.config.segment_weight > 0:
            weights.append(self.config.segment_weight)
            components.append(segment_distance)
        if self.config.feature_weight > 0:
            weights.append(self.config.feature_weight)
            components.append(feature_distance)
        if self.config.tone_weight > 0 and (ipa_a.chinese_char_count > 0 or ipa_b.chinese_char_count > 0):
            weights.append(self.config.tone_weight)
            components.append(tone_distance)
        elif self.config.tone_weight > 0:
            # No tone evidence, ignore weight
            pass
        if self.config.stress_weight > 0 and (ipa_a.english_word_count > 0 or ipa_b.english_word_count > 0):
            weights.append(self.config.stress_weight)
            components.append(stress_distance)
        elif self.config.stress_weight > 0:
            pass

        if not weights:
            weights = [1.0]
            components = [segment_distance]

        weights = normalize_weights(weights)
        total = sum(w * c for w, c in zip(weights, components))

        return DistanceBreakdown(
            segment_distance=segment_distance,
            feature_distance=feature_distance,
            tone_distance=tone_distance,
            stress_distance=stress_distance,
            total=total,
        )

    def _segment_distance(self, ipa_a: IPAResult, ipa_b: IPAResult) -> float:
        sequence_a = " ".join(ipa_a.segments)
        sequence_b = " ".join(ipa_b.segments)
        scores: List[float] = []
        length = max(len(ipa_a.segments), len(ipa_b.segments), 1)
        for metric in self.config.segment_metrics:
            if metric == "phonetic_edit_distance":
                dist = self._ped.dist(sequence_a, sequence_b)
            elif metric == "aline":
                dist = self._aline.dist(sequence_a, sequence_b)
            else:  # pragma: no cover - guarded by validation
                raise ValueError(f"Unknown metric: {metric}")
            scores.append(dist / length)
        return float(sum(scores) / len(scores))

    def _feature_distance(self, ipa_a: IPAResult, ipa_b: IPAResult) -> float:
        seq_a = ipa_a.feature_vectors
        seq_b = ipa_b.feature_vectors
        if not seq_a and not seq_b:
            return 0.0
        if not seq_a or not seq_b:
            return 1.0
        dtw_cost = self._dtw(seq_a, seq_b)
        return dtw_cost

    def _tone_distance(self, ipa_a: IPAResult, ipa_b: IPAResult) -> float:
        if not ipa_a.tones and not ipa_b.tones:
            return 0.0
        penalty = self._edit_distance(ipa_a.tones, ipa_b.tones)
        denom = max(ipa_a.chinese_char_count, ipa_b.chinese_char_count, 1)
        return penalty / denom

    def _stress_distance(self, ipa_a: IPAResult, ipa_b: IPAResult) -> float:
        if not ipa_a.stress_pattern and not ipa_b.stress_pattern:
            return 0.0
        penalty = self._edit_distance(ipa_a.stress_pattern, ipa_b.stress_pattern)
        denom = max(ipa_a.english_word_count, ipa_b.english_word_count, 1)
        return penalty / denom

    def _dtw(self, seq_a: List[np.ndarray], seq_b: List[np.ndarray]) -> float:
        n, m = len(seq_a), len(seq_b)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._feature_cost(seq_a[i - 1], seq_b[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        path_length = n + m
        if path_length == 0:
            return 0.0
        return float(dp[n, m] / path_length)

    def _feature_cost(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.config.feature_metric == "euclidean":
            return float(np.linalg.norm(a - b))
        # cosine distance
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 1.0
        cosine = float(np.dot(a, b) / denom)
        return 0.5 * (1 - cosine)

    @staticmethod
    def _edit_distance(seq_a: Iterable[int], seq_b: Iterable[int]) -> int:
        a = list(seq_a)
        b = list(seq_b)
        if not a:
            return len(b)
        if not b:
            return len(a)
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            dp[i][0] = i
        for j in range(len(b) + 1):
            dp[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[-1][-1]
