"""Top-level package for Chinese ASR lexicon correction."""

from .config import CorrectionConfig, DistanceConfig, ToneDistanceConfig
from .distance import DistanceCalculator
from .matcher import CandidateTerm, LexiconCorrector, MatchResult
from .phonetics import PhoneticConverter, PhoneticSequence, Syllable

__all__ = [
    "CorrectionConfig",
    "DistanceConfig",
    "ToneDistanceConfig",
    "DistanceCalculator",
    "CandidateTerm",
    "LexiconCorrector",
    "MatchResult",
    "PhoneticConverter",
    "PhoneticSequence",
    "Syllable",
]
