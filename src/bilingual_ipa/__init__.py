"""Public API for bilingual IPA conversion."""
from .conversion import IPAConversionResult, LanguageSegmenter, text_to_ipa
from .distances import (
    AggregateStrategy,
    CompositeDistanceCalculator,
    DistanceCalculator,
    PhoneDistanceCallable,
    PhoneDistanceCalculator,
    ToneDistanceCalculator,
)
from .phonetic_search import WindowDistance, window_phonetic_distances

__all__ = [
    "text_to_ipa",
    "LanguageSegmenter",
    "IPAConversionResult",
    "AggregateStrategy",
    "DistanceCalculator",
    "PhoneDistanceCallable",
    "PhoneDistanceCalculator",
    "ToneDistanceCalculator",
    "CompositeDistanceCalculator",
    "WindowDistance",
    "window_phonetic_distances",
]
