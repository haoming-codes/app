"""Top-level package for phonetic correction utilities."""
from .config import DistanceConfig
from .phonetics import MultilingualPhonemizer, PhoneticTranscription
from .distances import DistanceResult, PhoneticDistanceCalculator
from .matching import PhoneticMatcher, MatchCandidate

__all__ = [
    "DistanceConfig",
    "MultilingualPhonemizer",
    "PhoneticTranscription",
    "PhoneticDistanceCalculator",
    "DistanceResult",
    "PhoneticMatcher",
    "MatchCandidate",
]
