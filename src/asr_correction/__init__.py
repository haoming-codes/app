"""Phonetic distance based correction utilities for ASR outputs."""

from .config import (
    DistanceComputationConfig,
    SegmentalMetricConfig,
    ToneDistanceConfig,
)
from .distance import compute_distance
from .corrector import KnowledgeBaseEntry, CorrectionResult, Corrector

__all__ = [
    "DistanceComputationConfig",
    "SegmentalMetricConfig",
    "ToneDistanceConfig",
    "compute_distance",
    "KnowledgeBaseEntry",
    "CorrectionResult",
    "Corrector",
]
