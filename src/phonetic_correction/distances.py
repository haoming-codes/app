"""Distance utilities for comparing phonetic transcriptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig
from .phonetics import PhoneticTranscription


@dataclass
class DistanceResult:
    """Breakdown of the composite phonetic distance."""

    segment_distance: float
    tone_distance: float
    stress_distance: float
    total_distance: float


class PhoneticDistanceCalculator:
    """Compute distances between phonetic transcriptions."""

    def __init__(self, config: DistanceConfig) -> None:
        self.config = config
        self._segment_metrics = {
            "phonetic_edit_distance": PhoneticEditDistance(),
            "aline": ALINE(),
        }

    def _segment_distance(self, first: PhoneticTranscription, second: PhoneticTranscription) -> float:
        weights = self.config.normalized_weights()
        scores: List[float] = []
        if not first.ipa and not second.ipa:
            return 0.0
        for name, weight in weights.items():
            if weight <= 0:
                continue
            if name == "dtw":
                score = _normalized_dtw(first.feature_vectors, second.feature_vectors)
            else:
                metric = self._segment_metrics.get(name)
                if metric is None:
                    continue
                score = metric.dist(first.ipa, second.ipa)
            scores.append(weight * score)
        if not scores:
            return 0.0
        return float(sum(scores))

    def _tone_distance(self, first: PhoneticTranscription, second: PhoneticTranscription) -> float:
        if first.tone_unit_count == 0 and second.tone_unit_count == 0:
            return 0.0
        norm = max(first.tone_unit_count, second.tone_unit_count, 1)
        cost = _edit_distance(first.tone_sequence, second.tone_sequence, mismatch_cost=1.0)
        return self.config.tone_penalty * (cost / norm)

    def _stress_distance(self, first: PhoneticTranscription, second: PhoneticTranscription) -> float:
        if first.stress_unit_count == 0 and second.stress_unit_count == 0:
            return 0.0
        norm = max(first.stress_unit_count, second.stress_unit_count, 1)
        cost = _edit_distance(first.stress_sequence, second.stress_sequence, mismatch_cost=1.0)
        return self.config.stress_penalty * (cost / norm)

    def distance(self, first: PhoneticTranscription, second: PhoneticTranscription) -> DistanceResult:
        """Compute the composite phonetic distance between two strings."""

        lambdas = self.config.normalized_lambdas()
        segment = self._segment_distance(first, second)
        tone = self._tone_distance(first, second)
        stress = self._stress_distance(first, second)
        total = (
            lambdas["segment"] * segment
            + lambdas["tone"] * tone
            + lambdas["stress"] * stress
        )
        return DistanceResult(segment, tone, stress, total)


def _normalized_dtw(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0
    cost_matrix = _dtw_cost(first, second)
    path_cost = cost_matrix[-1, -1]
    path_length = _traceback_length(cost_matrix)
    average_cost = path_cost / max(path_length, 1)
    return average_cost / (1.0 + average_cost)


def _dtw_cost(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> np.ndarray:
    n, m = len(first), len(second)
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = np.linalg.norm(first[i - 1] - second[j - 1])
            cost[i, j] = diff + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost


def _traceback_length(cost: np.ndarray) -> int:
    i, j = cost.shape[0] - 1, cost.shape[1] - 1
    length = 0
    while i > 0 and j > 0:
        prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        if prev == cost[i - 1, j - 1]:
            i -= 1
            j -= 1
        elif prev == cost[i - 1, j]:
            i -= 1
        else:
            j -= 1
        length += 1
    length += i + j
    return length if length > 0 else 1


def _edit_distance(first: Sequence[int], second: Sequence[int], mismatch_cost: float = 1.0) -> float:
    n, m = len(first), len(second)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i, 0] = i * mismatch_cost
    for j in range(1, m + 1):
        dp[0, j] = j * mismatch_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = 0.0 if first[i - 1] == second[j - 1] else mismatch_cost
            dp[i, j] = min(
                dp[i - 1, j] + mismatch_cost,
                dp[i, j - 1] + mismatch_cost,
                dp[i - 1, j - 1] + substitution,
            )
    return float(dp[n, m])
