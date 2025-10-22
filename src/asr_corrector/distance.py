"""Phonetic distance computation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import FeatureMetric, PhoneticScoringConfig
from .converter import MultilingualPhoneticConverter, PhoneticSequence
from .metrics import NormalizedEditDistance


def _feature_distance(vec_a: Sequence[int], vec_b: Sequence[int], metric: FeatureMetric) -> float:
    diff = np.abs(np.asarray(vec_a, dtype=float) - np.asarray(vec_b, dtype=float))
    if metric == "l2":
        return float(np.linalg.norm(diff) / (2 * math.sqrt(len(diff))))
    return float(np.sum(diff) / (2 * len(diff)))


def _dtw_cost(
    seq_a: Sequence[Sequence[int]],
    seq_b: Sequence[Sequence[int]],
    metric: FeatureMetric,
) -> float:
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 or len_b == 0:
        return 0.0
    cost_matrix = np.full((len_a + 1, len_b + 1), np.inf)
    steps_matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    cost_matrix[0, 0] = 0.0

    for i in range(len_a):
        for j in range(len_b):
            local = _feature_distance(seq_a[i], seq_b[j], metric)
            for di, dj in ((1, 1), (1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                new_cost = cost_matrix[i, j] + local
                new_steps = steps_matrix[i, j] + 1
                if new_cost < cost_matrix[ni, nj] or (
                    math.isclose(new_cost, cost_matrix[ni, nj]) and new_steps < steps_matrix[ni, nj]
                ):
                    cost_matrix[ni, nj] = new_cost
                    steps_matrix[ni, nj] = new_steps

    total_steps = steps_matrix[len_a, len_b]
    if total_steps == 0:
        return 0.0
    return float(cost_matrix[len_a, len_b] / total_steps)


@dataclass(slots=True)
class PhoneticDistanceCalculator:
    """Compute combined phonetic distance between two strings."""

    config: PhoneticScoringConfig
    converter: MultilingualPhoneticConverter
    segment_metric: ALINE | PhoneticEditDistance
    tone_distance: NormalizedEditDistance
    stress_distance: NormalizedEditDistance

    def __init__(
        self,
        config: PhoneticScoringConfig | None = None,
        converter: MultilingualPhoneticConverter | None = None,
    ) -> None:
        self.config = config or PhoneticScoringConfig()
        self.converter = converter or MultilingualPhoneticConverter(
            tone_neutral=self.config.tone_neutral
        )
        if self.config.segment_metric == "phonetic_edit":
            self.segment_metric = PhoneticEditDistance(**(self.config.segment_metric_kwargs or {}))
        else:
            self.segment_metric = ALINE(**(self.config.segment_metric_kwargs or {}))
        self.tone_distance = NormalizedEditDistance()
        self.stress_distance = NormalizedEditDistance()

    def distance(self, text_a: str, text_b: str) -> float:
        seq_a = self.converter.to_sequence(text_a)
        seq_b = self.converter.to_sequence(text_b)
        return self._distance_sequences(seq_a, seq_b)

    def _distance_sequences(self, seq_a: PhoneticSequence, seq_b: PhoneticSequence) -> float:
        w_segment, w_dtw, w_tone, w_stress = self.config.normalized_weights()
        components: List[float] = []
        weights: List[float] = []

        if w_segment > 0:
            seg = self.segment_metric.dist(seq_a.ipa, seq_b.ipa)
            norm = 0.0
            denom = max(len(seq_a.ipa), len(seq_b.ipa))
            if denom:
                norm = seg / denom
            components.append(norm)
            weights.append(w_segment)

        if w_dtw > 0:
            dtw = _dtw_cost(seq_a.features, seq_b.features, self.config.feature_metric)
            components.append(dtw)
            weights.append(w_dtw)

        if w_tone > 0:
            tone = self.tone_distance(seq_a.tone_sequence, seq_b.tone_sequence)
            components.append(tone)
            weights.append(w_tone)

        if w_stress > 0:
            stress = self.stress_distance(seq_a.stress_sequence, seq_b.stress_sequence)
            components.append(stress)
            weights.append(w_stress)

        if not components:
            return 0.0
        weighted = sum(weight * component for weight, component in zip(weights, components))
        return weighted

    def components(self, text_a: str, text_b: str) -> dict[str, float]:
        seq_a = self.converter.to_sequence(text_a)
        seq_b = self.converter.to_sequence(text_b)
        return {
            "segment": self.segment_metric.dist(seq_a.ipa, seq_b.ipa),
            "dtw": _dtw_cost(seq_a.features, seq_b.features, self.config.feature_metric),
            "tone": self.tone_distance(seq_a.tone_sequence, seq_b.tone_sequence),
            "stress": self.stress_distance(seq_a.stress_sequence, seq_b.stress_sequence),
        }
