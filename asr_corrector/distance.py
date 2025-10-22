from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig, SegmentMetricConfig
from .phonetics import PhoneticRepresentation, PhoneticTranscriber


@dataclass
class DistanceBreakdown:
    total: float
    segment: float
    feature: float
    metrics: Dict[str, float]
    tone: float
    stress: float


class DistanceCalculator:
    """Compute combined phonetic distances according to a configuration."""

    def __init__(self, config: DistanceConfig | None = None, transcriber: PhoneticTranscriber | None = None) -> None:
        self.config = config or DistanceConfig()
        self.transcriber = transcriber or PhoneticTranscriber()
        self._metric_instances = {
            cfg.name: self._instantiate_metric(cfg)
            for cfg in self.config.segment.metrics
        }

    def compute(self, first: str, second: str) -> DistanceBreakdown:
        rep_a = self.transcriber.transcribe(first)
        rep_b = self.transcriber.transcribe(second)

        metric_scores = self._metric_scores(rep_a, rep_b)
        feature_score = self._feature_distance(rep_a, rep_b)
        segment_score = self._combine_segment_scores(metric_scores, feature_score)
        tone_score = self._sequence_distance(rep_a.tones, rep_b.tones, component="tone")
        stress_score = self._sequence_distance(rep_a.stresses, rep_b.stresses, component="stress")

        weights = self.config.weights
        total_weight = weights.segment + weights.tone + weights.stress
        if total_weight == 0:
            total = 0.0
        else:
            total = (
                segment_score * weights.segment
                + tone_score * weights.tone
                + stress_score * weights.stress
            ) / total_weight

        return DistanceBreakdown(
            total=total,
            segment=segment_score,
            feature=feature_score,
            metrics=metric_scores,
            tone=tone_score,
            stress=stress_score,
        )

    def _metric_scores(self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for cfg in self.config.segment.metrics:
            metric = self._metric_instances[cfg.name]
            dist = metric.dist(rep_a.ipa, rep_b.ipa)
            scores[cfg.name] = float(dist)
        return scores

    def _feature_distance(self, rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> float:
        if not rep_a.features or not rep_b.features:
            return 0.0
        cost, steps = _dtw(rep_a.features, rep_b.features, metric=self.config.segment.feature_distance)
        if steps == 0:
            return 0.0
        return cost / steps

    def _combine_segment_scores(self, metric_scores: Dict[str, float], feature_score: float) -> float:
        metrics_weight = self.config.segment.metrics_weight
        feature_weight = self.config.segment.feature_weight
        if metrics_weight <= 0 and feature_weight <= 0:
            return 0.0

        metrics_total = sum(cfg.weight * metric_scores[cfg.name] for cfg in self.config.segment.metrics)
        weight_sum = sum(cfg.weight for cfg in self.config.segment.metrics)
        metrics_average = metrics_total / weight_sum if weight_sum else 0.0

        if feature_weight <= 0:
            return metrics_average
        if metrics_weight <= 0:
            return feature_score

        return (
            metrics_average * metrics_weight + feature_score * feature_weight
        ) / (metrics_weight + feature_weight)

    def _sequence_distance(self, seq_a: List[int], seq_b: List[int], *, component: str) -> float:
        if not seq_a and not seq_b:
            return 0.0
        if self.config.tone_alignment != "levenshtein" and component == "tone":
            raise ValueError(f"Unsupported tone alignment {self.config.tone_alignment}")
        if self.config.stress_alignment != "levenshtein" and component == "stress":
            raise ValueError(f"Unsupported stress alignment {self.config.stress_alignment}")
        distance = _levenshtein(seq_a, seq_b)
        length = max(len(seq_a), len(seq_b), 1)
        return distance / length

    def _instantiate_metric(self, cfg: SegmentMetricConfig):
        if cfg.factory is not None:
            return cfg.factory()
        if cfg.name == "phonetic_edit_distance":
            return PhoneticEditDistance()
        if cfg.name == "aline":
            return ALINE()
        raise ValueError(f"Unsupported metric {cfg.name}")


def _dtw(seq_a: Iterable[List[int]], seq_b: Iterable[List[int]], metric: str = "l1") -> tuple[float, int]:
    a = list(seq_a)
    b = list(seq_b)
    if not a or not b:
        return 0.0, 0
    m = len(a)
    n = len(b)
    costs = [[math.inf] * (n + 1) for _ in range(m + 1)]
    steps = [[0] * (n + 1) for _ in range(m + 1)]
    costs[0][0] = 0.0
    steps[0][0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = _vector_distance(a[i - 1], b[j - 1], metric)
            candidates = (
                (costs[i - 1][j], steps[i - 1][j]),
                (costs[i][j - 1], steps[i][j - 1]),
                (costs[i - 1][j - 1], steps[i - 1][j - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda x: (x[0], x[1]))
            costs[i][j] = best_cost + cost
            steps[i][j] = best_steps + 1
    return costs[m][n], steps[m][n]


def _vector_distance(vec_a: List[int], vec_b: List[int], metric: str) -> float:
    if metric == "l1":
        return sum(abs(x - y) for x, y in zip(vec_a, vec_b)) / len(vec_a)
    if metric == "l2":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec_a, vec_b))) / math.sqrt(len(vec_a))
    raise ValueError(f"Unsupported feature distance metric {metric}")


def _levenshtein(seq_a: List[int], seq_b: List[int]) -> int:
    m = len(seq_a)
    n = len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


__all__ = ["DistanceCalculator", "DistanceBreakdown"]
