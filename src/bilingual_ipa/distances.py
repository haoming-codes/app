"""Distance calculators for IPA conversion results."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Callable, Literal

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance
from sktime.distances import twe_distance

from .conversion import IPAConversionResult

PhoneDistanceCallable = Callable[[str, str], float]
AggregateStrategy = Literal["mean", "sum", "min", "max"]


class DistanceCalculator(ABC):
    """Abstract base class for IPA distance calculators."""

    @abstractmethod
    def distance(self, left: IPAConversionResult, right: IPAConversionResult) -> float:
        """Return the distance between two conversion results."""


class PhoneDistanceCalculator(DistanceCalculator):
    """Distance calculator operating on concatenated IPA phones."""

    _phonetic_edit_distance = PhoneticEditDistance()
    _aline_distance = ALINE()
    _panphon_distance = PanphonDistance()

    #: Mapping of available metric identifiers to callables that compute distances.
    AVAILABLE_METRICS: dict[str, PhoneDistanceCallable] = {
        "phonetic_edit_distance": _phonetic_edit_distance.dist,
        "aline": _aline_distance.dist,
        "feature_edit_distance_div_maxlen": _panphon_distance.feature_edit_distance_div_maxlen,
        "hamming_feature_edit_distance_div_maxlen": _panphon_distance.hamming_feature_edit_distance_div_maxlen,
        "weighted_feature_edit_distance_div_maxlen": _panphon_distance.weighted_feature_edit_distance_div_maxlen,
    }

    def __init__(
        self,
        metrics: Iterable[str] | str | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        aggregate: AggregateStrategy = "mean",
    ) -> None:
        if metrics is None:
            metric_names = list(self.AVAILABLE_METRICS)
        elif isinstance(metrics, str):
            metric_names = [metrics]
        else:
            metric_names = list(metrics)

        for name in metric_names:
            if name not in self.AVAILABLE_METRICS:
                raise KeyError(f"Unknown distance metric: {name}")

        self._metrics = metric_names
        self._weights = dict(weights or {})
        self._aggregate = aggregate

    def _combine(self, distances: Sequence[float]) -> float:
        if not distances:
            raise ValueError("At least one distance metric must be provided for combination.")

        if self._aggregate in {"min", "max"}:
            op = min if self._aggregate == "min" else max
            return op(distances)

        weights = [self._weights.get(name, 1.0) for name in self._metrics]
        weighted = [value * weight for value, weight in zip(distances, weights, strict=True)]

        if self._aggregate == "sum":
            return float(sum(weighted))

        if self._aggregate == "mean":
            total_weight = float(sum(weights))
            if total_weight == 0:
                raise ValueError("The sum of weights must not be zero when computing the mean.")
            return float(sum(weighted) / total_weight)

        raise ValueError(f"Unsupported aggregate strategy: {self._aggregate}")

    def distance(self, left: IPAConversionResult, right: IPAConversionResult) -> float:
        phone_left = "".join(left.phones)
        phone_right = "".join(right.phones)

        if not phone_left and not phone_right:
            return 0.0

        distances = [
            self.AVAILABLE_METRICS[name](phone_left, phone_right) for name in self._metrics
        ]
        return self._combine(distances)


class ToneDistanceCalculator(DistanceCalculator):
    """Distance calculator operating on tone mark sequences."""

    def __init__(self) -> None:
        self._tone_indices: dict[str, float] = {"": 0.0}

    def _encode_tones(self, tones: Sequence[str]) -> np.ndarray:
        if not tones:
            return np.array([], dtype=float)
        encoded = []
        for mark in tones:
            value = self._tone_indices.get(mark)
            if value is None:
                value = float(len(self._tone_indices))
                self._tone_indices[mark] = value
            encoded.append(value)
        return np.array(encoded, dtype=float)

    def distance(self, left: IPAConversionResult, right: IPAConversionResult) -> float:
        tones_left = self._encode_tones(left.tone_marks)
        tones_right = self._encode_tones(right.tone_marks)

        if tones_left.size == 0 and tones_right.size == 0:
            return 0.0
        if tones_left.size == 0:
            return float(tones_right.size)
        if tones_right.size == 0:
            return float(tones_left.size)

        return float(twe_distance(tones_left, tones_right))/(len(tones_left)+len(tones_right))


class CompositeDistanceCalculator(DistanceCalculator):
    """Combine multiple distance calculators into a composite metric."""

    def __init__(
        self,
        calculators: Iterable[DistanceCalculator],
        *,
        weights: Sequence[float] | None = None,
        aggregate: AggregateStrategy = "mean",
    ) -> None:
        self._calculators = tuple(calculators)
        if not self._calculators:
            raise ValueError("At least one calculator must be supplied.")

        if weights is None:
            self._weights = tuple(1.0 for _ in self._calculators)
        else:
            weights = tuple(weights)
            if len(weights) != len(self._calculators):
                raise ValueError("weights must match the number of calculators")
            self._weights = weights

        self._aggregate = aggregate

    def distance(self, left: IPAConversionResult, right: IPAConversionResult) -> float:
        distances = [calculator.distance(left, right) for calculator in self._calculators]

        if self._aggregate in {"min", "max"}:
            op = min if self._aggregate == "min" else max
            return op(distances)

        weighted = [value * weight for value, weight in zip(distances, self._weights, strict=True)]

        if self._aggregate == "sum":
            return float(sum(weighted))

        if self._aggregate == "mean":
            total_weight = float(sum(self._weights))
            if total_weight == 0:
                raise ValueError("The sum of weights must not be zero when computing the mean.")
            return float(sum(weighted) / total_weight)

        raise ValueError(f"Unsupported aggregate strategy: {self._aggregate}")


__all__ = [
    "AggregateStrategy",
    "CompositeDistanceCalculator",
    "DistanceCalculator",
    "PhoneDistanceCallable",
    "PhoneDistanceCalculator",
    "ToneDistanceCalculator",
]
