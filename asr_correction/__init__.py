"""Tools for correcting ASR outputs using phonetic distances."""

from .config import CorrectionConfig, DistanceComponentConfig
from .distances import PhoneticDistanceCalculator, compute_distance
from .knowledge import KnowledgeBase, KnowledgeBaseEntry
from .matcher import CorrectionEngine, CorrectionSuggestion

__all__ = [
    "CorrectionConfig",
    "DistanceComponentConfig",
    "PhoneticDistanceCalculator",
    "compute_distance",
    "KnowledgeBase",
    "KnowledgeBaseEntry",
    "CorrectionEngine",
    "CorrectionSuggestion",
]
