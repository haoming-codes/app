"""Top-level package exports for phonetic_rag."""

from .config import DistanceAggregationConfig, PhoneticPipelineConfig
from .distance import PhoneticDistanceCalculator, SegmentDistanceBreakdown
from .phonetics import PhoneticRepresentation, PhoneticTranscriber
from .corrector import CandidateMatch, PhoneticCorrector

__all__ = [
    "DistanceAggregationConfig",
    "PhoneticPipelineConfig",
    "PhoneticDistanceCalculator",
    "SegmentDistanceBreakdown",
    "PhoneticRepresentation",
    "PhoneticTranscriber",
    "CandidateMatch",
    "PhoneticCorrector",
]
