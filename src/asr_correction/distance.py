"""Phonetic distance calculations for multilingual ASR correction."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance

from .config import DistanceConfig
from .phonetics import IPAConverter, PhoneticSequence, Phone


@dataclass(slots=True)
class DistanceBreakdown:
    """Detailed view of the combined phonetic distance."""

    segment: float
    features: float
    tone: float
    stress: float
    overall: float
    left: PhoneticSequence
    right: PhoneticSequence


class DistanceCalculator:
    """Compute phonetic distances between two strings."""

    def __init__(
        self,
        config: Optional[DistanceConfig] = None,
        converter: Optional[IPAConverter] = None,
    ) -> None:
        self.config = config or DistanceConfig()
        self.converter = converter or IPAConverter()
        self._segment_metric = self._build_segment_metric(self.config.segment_metric)
        self._feature_distance = self._build_feature_distance(self.config.feature_distance)

    def distance(self, left: str, right: str) -> DistanceBreakdown:
        """Return the weighted phonetic distance between ``left`` and ``right``."""

        left_seq = self.converter.sequence(left)
        right_seq = self.converter.sequence(right)
        segment_cost = self._segment_distance(left_seq, right_seq)
        feature_cost, tone_cost, stress_cost = self._feature_based_distances(
            left_seq.phones, right_seq.phones
        )
        overall = (
            self.config.lambda_segment * segment_cost
            + self.config.lambda_features * feature_cost
            + self.config.lambda_tone * tone_cost
            + self.config.lambda_stress * stress_cost
        )
        return DistanceBreakdown(
            segment=segment_cost,
            features=feature_cost,
            tone=tone_cost,
            stress=stress_cost,
            overall=overall,
            left=left_seq,
            right=right_seq,
        )

    # ------------------------------------------------------------------
    # Segment distance
    # ------------------------------------------------------------------
    def _segment_distance(self, left: PhoneticSequence, right: PhoneticSequence) -> float:
        if not left.ipa and not right.ipa:
            return 0.0
        raw = self._segment_metric.dist(left.ipa, right.ipa)
        length = max(len(left.phones), len(right.phones), 1)
        return raw / length

    @staticmethod
    def _build_segment_metric(name: str):
        if name == "phonetic_edit":
            return PhoneticEditDistance()
        if name == "aline":
            return ALINE()
        raise ValueError(f"Unknown segment metric: {name}")

    # ------------------------------------------------------------------
    # Feature alignment + suprasegmental penalties
    # ------------------------------------------------------------------
    def _build_feature_distance(self, name: str) -> Callable[[np.ndarray, np.ndarray], float]:
        if name == "l1":
            return lambda a, b: float(np.sum(np.abs(a - b)) / (2.0 * max(len(a), 1)))
        if name == "l2":
            size = None

            def _l2(a: np.ndarray, b: np.ndarray) -> float:
                nonlocal size
                if size is None:
                    size = len(a)
                return float(np.linalg.norm(a - b) / (2.0 * math.sqrt(size)))

            return _l2
        if name == "cosine":
            def _cos(a: np.ndarray, b: np.ndarray) -> float:
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return 0.0
                cos = float(np.dot(a, b) / denom)
                return 0.5 * (1.0 - cos)

            return _cos
        raise ValueError(f"Unknown feature distance: {name}")

    def _feature_based_distances(
        self, left: Sequence[Phone], right: Sequence[Phone]
    ) -> Tuple[float, float, float]:
        if not left and not right:
            return 0.0, 0.0, 0.0
        if not left or not right:
            feature_cost = 1.0
            tone_cost = 1.0 if (left and any(p.tone for p in left)) or (right and any(p.tone for p in right)) else 0.0
            stress_cost = 1.0 if (left and any(p.stress for p in left)) or (right and any(p.stress for p in right)) else 0.0
            return feature_cost, tone_cost, stress_cost
        total_cost, path = _dtw(left, right, self._feature_distance)
        average_cost = total_cost / max(len(path), 1)
        tone_cost = _suprasegmental_penalty(
            left,
            right,
            path,
            attribute="tone",
            penalty=self.config.tone_penalty,
            fallback=self.config.language_switch_penalty,
        )
        stress_cost = _suprasegmental_penalty(
            left,
            right,
            path,
            attribute="stress",
            penalty=self.config.stress_penalty,
            fallback=None,
        )
        return average_cost, tone_cost, stress_cost


def _dtw(
    left: Sequence[Phone],
    right: Sequence[Phone],
    distance: Callable[[np.ndarray, np.ndarray], float],
) -> Tuple[float, List[Tuple[int, int]]]:
    n = len(left)
    m = len(right)
    cost = np.full((n, m), np.inf, dtype=float)
    back = np.zeros((n, m), dtype=np.int8)

    cost[0, 0] = distance(left[0].features, right[0].features)
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + distance(left[i].features, right[0].features)
        back[i, 0] = 0  # came from up
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + distance(left[0].features, right[j].features)
        back[0, j] = 1  # came from left

    for i in range(1, n):
        for j in range(1, m):
            candidates = (
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
            move = int(np.argmin(candidates))
            back[i, j] = move
            cost[i, j] = distance(left[i].features, right[j].features) + candidates[move]

    path: List[Tuple[int, int]] = [(n - 1, m - 1)]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        move = back[i, j]
        if move == 2 and i > 0 and j > 0:
            i -= 1
            j -= 1
        elif move == 0 and i > 0:
            i -= 1
        elif move == 1 and j > 0:
            j -= 1
        else:  # fallback when at the boundary
            if i > 0:
                i -= 1
            elif j > 0:
                j -= 1
        path.append((i, j))
    path.reverse()
    return float(cost[n - 1, m - 1]), path


def _suprasegmental_penalty(
    left: Sequence[Phone],
    right: Sequence[Phone],
    path: Sequence[Tuple[int, int]],
    *,
    attribute: str,
    penalty: float,
    fallback: Optional[float],
) -> float:
    comparisons = 0
    mismatches = 0
    for i, j in path:
        value_left = getattr(left[i], attribute)
        value_right = getattr(right[j], attribute)
        if value_left is None and value_right is None:
            continue
        if value_left is None or value_right is None:
            mismatches += 1
        elif value_left != value_right:
            mismatches += 1
        comparisons += 1
    if comparisons == 0:
        if fallback is not None:
            left_has_attr = any(getattr(p, attribute) for p in left)
            right_has_attr = any(getattr(p, attribute) for p in right)
            if left_has_attr != right_has_attr:
                return fallback
        return 0.0
    return (mismatches / comparisons) * penalty


__all__ = ["DistanceCalculator", "DistanceBreakdown"]
