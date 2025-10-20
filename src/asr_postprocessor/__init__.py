"""ASR post-processing utilities for correcting Mandarin name entity errors."""

from .correction import ChineseASRCorrector, CorrectionResult
from .distance import (
    CombinedDistanceConfig,
    PhoneticDistanceCalculator,
    SegmentalDistanceConfig,
    ToneDistanceCalculator,
    ToneDistanceConfig,
)
from .matcher import CandidateMatcher, CandidateTerm, MatchResult, MatcherConfig
from .transcription import PinyinConverter, PinyinConversionError, Syllable

__all__ = [
    "CandidateMatcher",
    "CandidateTerm",
    "ChineseASRCorrector",
    "CombinedDistanceConfig",
    "CorrectionResult",
    "MatchResult",
    "MatcherConfig",
    "PhoneticDistanceCalculator",
    "PinyinConversionError",
    "PinyinConverter",
    "SegmentalDistanceConfig",
    "Syllable",
    "ToneDistanceCalculator",
    "ToneDistanceConfig",
]
