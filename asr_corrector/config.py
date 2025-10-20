"""Configuration objects for ASR post-correction."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DistanceMetric(str, Enum):
    """Available segmental distance metrics."""

    ALINE = "aline"
    PANPHON = "panphon"


@dataclass
class CorrectionConfig:
    """Hyperparameters controlling post-correction behaviour."""

    threshold: float = 1.5
    """Maximum total distance allowed to trigger a correction."""

    lambda_tone: float = 0.5
    """Trade-off coefficient for the tone distance contribution."""

    distance_metric: DistanceMetric = DistanceMetric.PANPHON
    """Segmental distance metric to use."""

    segmental_weight: float = 1.0
    """Weight applied to the segmental distance before combining with tone distance."""

    tone_mismatch_penalty: float = 1.0
    """Penalty applied per unmatched tone when one sequence is longer than the other."""

    max_window_size: int | None = None
    """Maximum number of characters to consider for a candidate span.

    Defaults to the length of the longest lexicon entry when ``None``.
    """
