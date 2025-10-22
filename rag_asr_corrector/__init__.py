"""Tools for phonetic post-correction of multilingual ASR output."""

from .config import DistanceConfig, CorrectionConfig
from .ipa import MultilingualIPAConverter, IPAResult
from .distances import DistanceCalculator, DistanceBreakdown
from .knowledge_base import KnowledgeBase, KnowledgeBaseEntry
from .corrector import PhoneticCorrector, CorrectionSuggestion

__all__ = [
    "DistanceConfig",
    "CorrectionConfig",
    "MultilingualIPAConverter",
    "IPAResult",
    "DistanceCalculator",
    "DistanceBreakdown",
    "KnowledgeBase",
    "KnowledgeBaseEntry",
    "PhoneticCorrector",
    "CorrectionSuggestion",
]
