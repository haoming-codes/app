"""Phone distance utilities backed by abydos and PanPhon."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable, Literal

from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance

PhoneDistanceCallable = Callable[[str, str], float]

# Instantiate the metric providers once to avoid the overhead of repeated setup.
_phonetic_edit_distance = PhoneticEditDistance()
_aline_distance = ALINE()
_panphon_distance = PanphonDistance()

#: Mapping of available metric identifiers to callables that compute distances.
AVAILABLE_DISTANCE_METRICS: dict[str, PhoneDistanceCallable] = {
    "phonetic_edit_distance": _phonetic_edit_distance.dist,
    "aline": _aline_distance.dist,
    "feature_edit_distance_div_maxlen": _panphon_distance.feature_edit_distance_div_maxlen,
    "hamming_feature_edit_distance_div_maxlen": _panphon_distance.hamming_feature_edit_distance_div_maxlen,
    "weighted_feature_edit_distance_div_maxlen": _panphon_distance.weighted_feature_edit_distance_div_maxlen,
}

AggregateStrategy = Literal["mean", "sum", "min", "max"]


def compute_distances(
    phone_a: str,
    phone_b: str,
    metrics: Iterable[str] | str | None = None,
) -> dict[str, float]:
    """Compute the distance for each selected metric.

    Args:
        phone_a: The first IPA phone string.
        phone_b: The second IPA phone string.
        metrics: Metric names to evaluate. If omitted, all available metrics are
            used. A single string can be provided as a shortcut for a
            one-element collection.

    Returns:
        A mapping from metric name to the computed distance value.

    Raises:
        KeyError: If any of the requested metrics are not supported.
    """

    if metrics is None:
        metric_names = list(AVAILABLE_DISTANCE_METRICS)
    elif isinstance(metrics, str):
        metric_names = [metrics]
    else:
        metric_names = list(metrics)

    distances: dict[str, float] = {}
    for name in metric_names:
        try:
            metric_callable = AVAILABLE_DISTANCE_METRICS[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Unknown distance metric: {name}") from exc
        distances[name] = metric_callable(phone_a, phone_b)
    return distances


def combine_distances(
    distances: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    aggregate: AggregateStrategy = "mean",
) -> float:
    """Combine a mapping of distances into a single value.

    Args:
        distances: Distance scores keyed by metric name.
        weights: Optional weights to apply to individual metrics. Any metric not
            present in the weights mapping defaults to a weight of ``1.0``.
        aggregate: Strategy for combining the weighted scores. Supported values
            are ``"mean"``, ``"sum"``, ``"min"``, and ``"max"``.

    Returns:
        The aggregated distance value.

    Raises:
        ValueError: If ``aggregate`` is not one of the supported strategies.
    """

    if not distances:
        raise ValueError("At least one distance metric must be provided for combination.")

    if aggregate in {"min", "max"}:
        op = min if aggregate == "min" else max
        return op(distances.values())

    weighted_scores = []
    total_weight = 0.0
    for name, value in distances.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        weighted_scores.append(value * weight)
        total_weight += weight

    if aggregate == "sum":
        return sum(weighted_scores)
    if aggregate == "mean":
        if total_weight == 0:
            raise ValueError("The sum of weights must not be zero when computing the mean.")
        return sum(weighted_scores) / total_weight

    raise ValueError(f"Unsupported aggregate strategy: {aggregate}")


def phone_distance(
    phone_a: str,
    phone_b: str,
    metrics: Iterable[str] | str | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    aggregate: AggregateStrategy = "mean",
) -> float:
    """Compute a combined distance between two IPA phones.

    This is a convenience helper that first computes the per-metric distances
    and then aggregates them according to ``aggregate``.

    Args:
        phone_a: The first IPA phone string.
        phone_b: The second IPA phone string.
        metrics: Metric names to evaluate. If omitted, all available metrics are
            used. A single string can be provided as a shortcut for a
            one-element collection.
        weights: Optional weights for individual metrics. See
            :func:`combine_distances` for details.
        aggregate: Aggregation strategy. See :func:`combine_distances` for
            options.

    Returns:
        The aggregated distance value.
    """

    distances = compute_distances(phone_a, phone_b, metrics)
    return combine_distances(distances, weights=weights, aggregate=aggregate)


__all__ = [
    "AVAILABLE_DISTANCE_METRICS",
    "AggregateStrategy",
    "PhoneDistanceCallable",
    "combine_distances",
    "compute_distances",
    "phone_distance",
]
