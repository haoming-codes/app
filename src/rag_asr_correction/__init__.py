"""RAG-inspired correction utilities for multilingual ASR outputs."""

from .config import (
    CorrectionConfig,
    SegmentalMetricConfig,
    SegmentalMetric,
    ToneDistanceConfig,
)
from .knowledge import KnowledgeBase, KnowledgeEntry
from .engine import CorrectionEngine, CorrectionResult, CorrectionCandidate
from .distance import DistanceComputer

__all__ = [
    "CorrectionConfig",
    "SegmentalMetricConfig",
    "SegmentalMetric",
    "ToneDistanceConfig",
    "KnowledgeBase",
    "KnowledgeEntry",
    "CorrectionEngine",
    "CorrectionResult",
    "CorrectionCandidate",
    "DistanceComputer",
]
