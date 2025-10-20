"""Distance metrics for phonetic comparison."""
from __future__ import annotations

from abc import ABC, abstractmethod


class DistanceMetric(ABC):
    """Interface for scoring similarity between phonetic strings."""

    @abstractmethod
    def distance(self, a: str, b: str) -> float:
        """Return a non-negative distance between ``a`` and ``b``."""


class PanphonDistance(DistanceMetric):
    """Distance metric backed by :mod:`panphon`."""

    def __init__(self):  # pragma: no cover - optional dependency
        try:
            from panphon.distance import Distance  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "PanphonDistance requires the optional 'phonetics' extra: "
                "pip install asr-corrector[phonetics]"
            ) from exc
        self._distance = Distance()

    def distance(self, a: str, b: str) -> float:  # pragma: no cover - optional dependency
        return self._distance.weighted_feature_edit_distance(a, b)


class LevenshteinDistance(DistanceMetric):
    """Simple Levenshtein distance for testing or fallback scenarios."""

    def distance(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        if not a:
            return float(len(b))
        if not b:
            return float(len(a))
        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current = [i]
            for j, char_b in enumerate(b, start=1):
                cost = 0 if char_a == char_b else 1
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + cost,
                    )
                )
            previous = current
        return float(previous[-1])


__all__ = ["DistanceMetric", "PanphonDistance", "LevenshteinDistance"]
