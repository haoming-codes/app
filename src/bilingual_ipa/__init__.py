"""Public API for bilingual IPA conversion."""
from .conversion import IPAConversionResult, LanguageSegmenter, text_to_ipa
from .contextual_correction import ASRContextualCorrector, build_correction_candidates
from .distances import (
    AggregateStrategy,
    CompositeDistanceCalculator,
    DistanceCalculator,
    PhoneDistanceCallable,
    PhoneDistanceCalculator,
    ToneDistanceCalculator,
)
from .phonetic_search import (
    PhoneticWindowRetriever,
    PhoneticWindowRewriter,
    WindowDistance,
    window_phonetic_distances,
)

__all__ = [
    "text_to_ipa",
    "LanguageSegmenter",
    "IPAConversionResult",
    "ASRContextualCorrector",
    "build_correction_candidates",
    "AggregateStrategy",
    "DistanceCalculator",
    "PhoneDistanceCallable",
    "PhoneDistanceCalculator",
    "ToneDistanceCalculator",
    "CompositeDistanceCalculator",
    "WindowDistance",
    "window_phonetic_distances",
    "PhoneticWindowRetriever",
    "PhoneticWindowRewriter",
]
