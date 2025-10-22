"""Distance calculations on phonetic representations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceAggregationConfig
from .phonetics import PhoneticRepresentation, PhoneticTranscriber


@dataclass
class SegmentDistanceBreakdown:
    """Detailed view of the combined phonetic distance."""

    total: float
    segment_component: float
    feature_component: float
    tone_component: float
    stress_component: float
    segment_metrics: Dict[str, float]


class PhoneticDistanceCalculator:
    """Compute aggregated phonetic distances between strings."""

    def __init__(
        self,
        config: DistanceAggregationConfig | None = None,
        *,
        transcriber: PhoneticTranscriber | None = None,
    ) -> None:
        self.config = config or DistanceAggregationConfig()
        self.transcriber = transcriber or PhoneticTranscriber()
        self._ped = PhoneticEditDistance()
        self._aline = ALINE()

    def distance(
        self,
        left: str,
        right: str,
        *,
        treat_all_caps_as_acronyms: bool = True,
    ) -> SegmentDistanceBreakdown:
        rep_left = self.transcriber.transcribe(
            left, treat_all_caps_as_acronyms=treat_all_caps_as_acronyms
        )
        rep_right = self.transcriber.transcribe(
            right, treat_all_caps_as_acronyms=treat_all_caps_as_acronyms
        )
        return self.distance_between_representations(rep_left, rep_right)

    def distance_between_representations(
        self, rep_left: PhoneticRepresentation, rep_right: PhoneticRepresentation
    ) -> SegmentDistanceBreakdown:
        cfg = self.config
        segment_metrics: Dict[str, float] = {}

        max_len = max(len(rep_left.phones), len(rep_right.phones), 1)

        normalized_weights = cfg.normalized_segment_weights()
        for metric_name, weight in normalized_weights.items():
            if metric_name == "phonetic_edit_distance":
                dist_abs = self._ped.dist_abs(rep_left.ipa, rep_right.ipa)
                segment_metrics[metric_name] = float(dist_abs / max_len)
            elif metric_name == "aline":
                segment_metrics[metric_name] = float(
                    self._aline.dist(rep_left.ipa, rep_right.ipa)
                )
            else:
                raise ValueError(f"Unsupported segment metric: {metric_name}")

        segment_component = sum(
            normalized_weights[metric_name] * value
            for metric_name, value in segment_metrics.items()
        )

        feature_norm = float(
            self._feature_distance(rep_left.feature_vectors, rep_right.feature_vectors)
        )
        if cfg.feature_weight > 0:
            segment_component = (
                segment_component + cfg.feature_weight * feature_norm
            ) / (1.0 + cfg.feature_weight)

        tone_component = float(self._tone_distance(rep_left, rep_right))
        stress_component = float(self._stress_distance(rep_left, rep_right))

        total = float(
            cfg.lambda_segment * segment_component
            + cfg.lambda_tone * tone_component * cfg.tone_weight
            + cfg.lambda_stress * stress_component * cfg.stress_weight
        )

        return SegmentDistanceBreakdown(
            total=float(total),
            segment_component=float(segment_component),
            feature_component=float(feature_norm),
            tone_component=float(tone_component),
            stress_component=float(stress_component),
            segment_metrics={k: float(v) for k, v in segment_metrics.items()},
        )

    def _tone_distance(
        self, rep_left: PhoneticRepresentation, rep_right: PhoneticRepresentation
    ) -> float:
        if not rep_left.tone_sequence and not rep_right.tone_sequence:
            return 0.0
        denom = max(
            rep_left.chinese_char_count,
            rep_right.chinese_char_count,
            len(rep_left.tone_sequence),
            len(rep_right.tone_sequence),
            1,
        )
        penalty = self._levenshtein(rep_left.tone_sequence, rep_right.tone_sequence)
        return penalty / denom

    def _stress_distance(
        self, rep_left: PhoneticRepresentation, rep_right: PhoneticRepresentation
    ) -> float:
        if not rep_left.stress_sequence and not rep_right.stress_sequence:
            return 0.0
        denom = max(rep_left.english_word_count, rep_right.english_word_count, 1)
        penalty = self._levenshtein(
            rep_left.stress_sequence, rep_right.stress_sequence
        )
        return penalty / denom

    def _feature_distance(
        self, features_left: List[List[float]], features_right: List[List[float]]
    ) -> float:
        if not features_left and not features_right:
            return 0.0
        if not features_left or not features_right:
            return 1.0
        cost, path_len = self._dtw_cost(features_left, features_right)
        if path_len == 0:
            return 0.0
        return cost / path_len

    def _dtw_cost(
        self, features_left: List[List[float]], features_right: List[List[float]]
    ) -> tuple[float, int]:
        n = len(features_left)
        m = len(features_right)
        cfg = self.config
        dist_matrix = np.full((n + 1, m + 1), math.inf, dtype=float)
        steps = np.zeros((n + 1, m + 1), dtype=int)
        dist_matrix[0, 0] = 0.0
        steps[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._local_feature_distance(
                    features_left[i - 1], features_right[j - 1]
                )
                candidates = [
                    (dist_matrix[i - 1, j], steps[i - 1, j]),
                    (dist_matrix[i, j - 1], steps[i, j - 1]),
                    (dist_matrix[i - 1, j - 1], steps[i - 1, j - 1]),
                ]
                prev_cost, prev_steps = min(candidates, key=lambda item: item[0])
                dist_matrix[i, j] = cost + prev_cost
                steps[i, j] = prev_steps + 1

        return float(dist_matrix[n, m]), int(steps[n, m])

    def _local_feature_distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        metric = self.config.dtw_local_metric
        if metric == "cosine":
            return 1.0 - self._cosine_similarity(left, right)
        if metric == "euclidean":
            return float(np.linalg.norm(np.array(left) - np.array(right)))
        if metric == "manhattan":
            return float(np.sum(np.abs(np.array(left) - np.array(right))))
        raise ValueError(f"Unsupported DTW local metric: {metric}")

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        left_vec = np.array(left)
        right_vec = np.array(right)
        denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        if denom == 0.0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / denom)

    @staticmethod
    def _levenshtein(seq1: Sequence, seq2: Sequence) -> int:
        len1 = len(seq1)
        len2 = len(seq2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[len1][len2]
