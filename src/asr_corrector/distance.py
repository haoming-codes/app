"""Distance computations for IPA and articulatory feature representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig
from .phonetics import IPAConverter, IPAResult


@dataclass
class SegmentDistanceResult:
    """Detailed components of the combined distance."""

    ipa_a: IPAResult
    ipa_b: IPAResult
    segment: float
    feature: float
    tone: float
    stress: float
    combined: float


class DistanceCalculator:
    """Compute combined phonetic distances between two strings."""

    def __init__(self, config: Optional[DistanceConfig] = None, converter: Optional[IPAConverter] = None) -> None:
        self.config = config or DistanceConfig()
        self.converter = converter or IPAConverter()
        self._phonetic_edit = PhoneticEditDistance()
        self._aline = ALINE()

    def compute(
        self,
        text_a: str,
        text_b: str,
        *,
        acronym_a: bool = False,
        acronym_b: bool = False,
        language_hint_a: Optional[str] = None,
        language_hint_b: Optional[str] = None,
    ) -> SegmentDistanceResult:
        ipa_a = self.converter.to_ipa(text_a, is_acronym=acronym_a, language_hint=language_hint_a)
        ipa_b = self.converter.to_ipa(text_b, is_acronym=acronym_b, language_hint=language_hint_b)

        segment_distance = self._segment_distance(ipa_a.phones, ipa_b.phones)
        feature_distance = self._feature_distance(ipa_a.feature_vectors, ipa_b.feature_vectors)
        tone_distance = self._suprasegmental_distance(ipa_a.tones, ipa_b.tones)
        stress_distance = self._suprasegmental_distance(ipa_a.stresses, ipa_b.stresses)

        combined = self._combine(segment_distance, feature_distance, tone_distance, stress_distance)
        return SegmentDistanceResult(
            ipa_a=ipa_a,
            ipa_b=ipa_b,
            segment=segment_distance,
            feature=feature_distance,
            tone=tone_distance,
            stress=stress_distance,
            combined=combined,
        )

    def _segment_distance(self, phones_a: list[str], phones_b: list[str]) -> float:
        if not phones_a and not phones_b:
            return 0.0
        metric = self.config.segment_metric
        joined_a = "".join(phones_a)
        joined_b = "".join(phones_b)
        if metric == "phonetic_edit":
            dist = self._phonetic_edit.dist(joined_a, joined_b)
        elif metric == "aline":
            dist = self._aline.dist(joined_a, joined_b)
        else:
            raise ValueError(f"Unsupported segment metric: {metric}")
        max_len = max(len(phones_a), len(phones_b), 1)
        return min(dist / max_len, 1.0)

    def _feature_distance(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        if features_a.size == 0 and features_b.size == 0:
            return 0.0
        if features_a.size == 0 or features_b.size == 0:
            return 1.0
        cost, steps = _dtw(features_a, features_b, metric=self.config.feature_metric)
        if steps == 0:
            return 0.0
        normalized = cost / steps
        return float(min(normalized, 1.0))

    def _suprasegmental_distance(self, seq_a: list[int], seq_b: list[int]) -> float:
        if not seq_a and not seq_b:
            return 0.0
        max_len = max(len(seq_a), len(seq_b), 1)
        dist = _levenshtein(seq_a, seq_b)
        return min(dist / max_len, 1.0)

    def _combine(self, segment: float, feature: float, tone: float, stress: float) -> float:
        cfg = self.config
        numerator = (
            cfg.segment_weight * segment
            + cfg.feature_weight * feature
            + cfg.tone_weight * tone
            + cfg.stress_weight * stress
        )
        denominator = cfg.segment_weight + cfg.feature_weight + cfg.tone_weight + cfg.stress_weight
        if denominator == 0:
            return 0.0
        return numerator / denominator


def _dtw(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> tuple[float, int]:
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = _feature_distance(a[i - 1], b[j - 1], metric)
            prev_choices = [
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
            ]
            idx = int(np.argmin([c for c, _ in prev_choices]))
            best_cost, best_steps = prev_choices[idx]
            cost[i, j] = local + best_cost
            steps[i, j] = best_steps + 1
    return float(cost[n, m]), int(steps[n, m])


def _feature_distance(vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0:
            return 1.0
        cos = np.dot(vec_a, vec_b) / denom
        return max(0.0, 1.0 - cos)
    if metric == "euclidean":
        return float(np.linalg.norm(vec_a - vec_b))
    raise ValueError(f"Unsupported feature metric: {metric}")


def _levenshtein(seq_a: list[int], seq_b: list[int]) -> int:
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])
