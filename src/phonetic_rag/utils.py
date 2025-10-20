"""Convenience helpers for quick distance calculations."""
from __future__ import annotations

from .config import DistanceConfig
from .distance import PhoneticDistanceCalculator


def compute_distance(a: str, b: str, config: DistanceConfig | None = None) -> float:
    """Compute combined phonetic distance between two substrings."""
    calculator = PhoneticDistanceCalculator(config)
    return calculator.distance(a, b)
