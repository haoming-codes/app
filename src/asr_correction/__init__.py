"""ASR correction utilities based on phonetic distances."""

from .config import DistanceConfig
from .distance import DistanceCalculator, DistanceBreakdown
from .phonetics import IPAConverter, PhoneticSequence, Phone
from .matcher import KnowledgeBaseMatcher, MatchResult

__all__ = [
    "DistanceConfig",
    "DistanceCalculator",
    "DistanceBreakdown",
    "IPAConverter",
    "PhoneticSequence",
    "Phone",
    "KnowledgeBaseMatcher",
    "MatchResult",
]
