"""Phonetic distance computation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig
from .features import PhoneticRepresentation, ipa_to_segments, segments_to_features
from .phonetics import ipa_transcription, stress_for_ipa, tones_for


@dataclass(slots=True)
class DistanceResult:
    """Holds the components of a phonetic distance comparison."""

    segment: float
    tone: float
    stress: float

    @property
    def combined(self) -> float:
        return self.segment + self.tone + self.stress


class PhoneticDistanceCalculator:
    """Compute phonetic distances between multilingual strings."""

    def __init__(self, config: DistanceConfig | None = None) -> None:
        self.config = config or DistanceConfig()
        self._phonetic_metric = PhoneticEditDistance()
        self._aline_metric = ALINE()

    def representation(self, text: str) -> PhoneticRepresentation:
        ipa = ipa_transcription(text)
        segments = ipa_to_segments(ipa)
        features = segments_to_features(segments)
        tones = tones_for(text)
        stresses = stress_for_ipa(ipa)
        return PhoneticRepresentation(ipa, segments, features, tones, stresses)

    def distance(self, left: str, right: str) -> DistanceResult:
        left_rep = self.representation(left)
        right_rep = self.representation(right)
        segment = self._segment_distance(left_rep, right_rep)
        tone = self._tone_distance(left_rep, right_rep)
        stress = self._stress_distance(left_rep, right_rep)
        return DistanceResult(segment, tone, stress)

    def _segment_distance(
        self, left: PhoneticRepresentation, right: PhoneticRepresentation
    ) -> float:
        distances: List[float] = []
        max_len = max(len(left.segments), len(right.segments)) or 1
        for metric in self.config.segment_metrics:
            if metric == "phonetic":
                d = self._phonetic_metric.dist(left.ipa, right.ipa)
            elif metric == "aline":
                d = self._aline_metric.dist(left.ipa, right.ipa)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            distances.append(d / max_len)
        if self.config.use_feature_dtw and left.features and right.features:
            feat = self._feature_distance(left, right)
            distances.append(feat / max_len)
        if not distances:
            return 0.0
        average = float(np.mean(distances))
        return average * self.config.segment_weight

    def _feature_distance(
        self, left: PhoneticRepresentation, right: PhoneticRepresentation
    ) -> float:
        left_mat = np.stack(left.features, axis=0)
        right_mat = np.stack(right.features, axis=0)
        return self._dtw(left_mat, right_mat, metric=self.config.feature_metric)

    def _tone_distance(
        self, left: PhoneticRepresentation, right: PhoneticRepresentation
    ) -> float:
        if not left.tones or not right.tones:
            return 0.0
        aligned = self._dtw(np.array(left.tones), np.array(right.tones), metric="manhattan")
        norm = max(len(left.tones), len(right.tones)) or 1
        return (aligned / norm) * self.config.tone_penalty * self.config.tone_weight

    def _stress_distance(
        self, left: PhoneticRepresentation, right: PhoneticRepresentation
    ) -> float:
        if not left.stresses or not right.stresses:
            return 0.0
        aligned = self._dtw(
            np.array(left.stresses), np.array(right.stresses), metric="manhattan"
        )
        norm = max(len(left.stresses), len(right.stresses)) or 1
        return (aligned / norm) * self.config.stress_penalty * self.config.stress_weight

    def _dtw(self, left: np.ndarray, right: np.ndarray, metric: str = "cosine") -> float:
        if left.size == 0 or right.size == 0:
            return 0.0
        left_seq = left.reshape(-1, 1) if left.ndim == 1 else left
        right_seq = right.reshape(-1, 1) if right.ndim == 1 else right
        n, m = left_seq.shape[0], right_seq.shape[0]
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = self._local_distance(left_seq[i - 1], right_seq[j - 1], metric)
                cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[n, m])

    def _local_distance(self, x: np.ndarray | float, y: np.ndarray | float, metric: str) -> float:
        if np.isscalar(x) and np.isscalar(y):
            return abs(float(x) - float(y))
        if metric == "cosine":
            numerator = float(np.dot(x, y))
            denom = float(np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
            sim = numerator / denom
            return 1.0 - sim
        if metric == "manhattan":
            return float(np.sum(np.abs(x - y)))
        if metric == "euclidean":
            return float(np.linalg.norm(x - y))
        raise ValueError(f"Unknown feature metric: {metric}")

    def combined_distance(self, left: str, right: str) -> float:
        return self.distance(left, right).combined


__all__ = [
    "PhoneticDistanceCalculator",
    "DistanceResult",
]
