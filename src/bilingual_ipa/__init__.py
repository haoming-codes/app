"""Public API for bilingual IPA conversion."""
from .converter import IPAConversionResult, LanguageSegmenter, text_to_ipa
from .phone_distance import (
    AVAILABLE_DISTANCE_METRICS,
    AggregateStrategy,
    PhoneDistanceCallable,
    combine_distances,
    compute_distances,
    phone_distance,
)
from .phonetic_search import WindowDistance, window_phonetic_distances

__all__ = [
    "text_to_ipa",
    "LanguageSegmenter",
    "IPAConversionResult",
    "AVAILABLE_DISTANCE_METRICS",
    "AggregateStrategy",
    "PhoneDistanceCallable",
    "combine_distances",
    "compute_distances",
    "phone_distance",
    "WindowDistance",
    "window_phonetic_distances",
]
