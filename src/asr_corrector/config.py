"""Configuration dataclasses for ASR correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


def _default_tone_confusion() -> Dict[int, Dict[int, float]]:
    """Return the default tone confusion cost matrix.

    The matrix entries represent the substitution cost between tones.
    Values were chosen heuristically based on typical Mandarin tone confusion
    patterns in ASR outputs: low cost for confusions within similar contour
    classes and higher cost for cross-class confusions.
    """

    base = {
        1: {1: 0.0, 2: 0.8, 3: 1.2, 4: 1.0, 5: 0.6},
        2: {1: 0.8, 2: 0.0, 3: 0.9, 4: 1.3, 5: 0.7},
        3: {1: 1.2, 2: 0.9, 3: 0.0, 4: 1.4, 5: 0.8},
        4: {1: 1.0, 2: 1.3, 3: 1.4, 4: 0.0, 5: 0.9},
        5: {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 0.0},
    }
    # Ensure symmetry.
    for tone, row in base.items():
        for other in list(row):
            row.setdefault(other, row[other])
            base.setdefault(other, {}).setdefault(tone, row[other])
    return base


@dataclass
class ToneDistanceConfig:
    """Configuration for tone-distance computation."""

    confusion_matrix: Mapping[int, Mapping[int, float]] = field(
        default_factory=_default_tone_confusion
    )
    insertion_penalty: float = 1.2
    deletion_penalty: Optional[float] = None


@dataclass
class DistanceConfig:
    """Configuration for the combined phonetic distance."""

    segmental_metric: str = "panphon"
    segmental_weight: float = 1.0
    tone_weight: float = 0.7
    segmental_kwargs: Mapping[str, Any] = field(default_factory=dict)
    tone: ToneDistanceConfig = field(default_factory=ToneDistanceConfig)


@dataclass
class CorrectionConfig:
    """Configuration for substring matching and correction."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    threshold: float = 6.0
    max_length_delta: int = 1
    enable_length_normalization: bool = True
