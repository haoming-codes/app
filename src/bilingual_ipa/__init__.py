"""Public API for bilingual IPA conversion."""
from .converter import LanguageSegmenter, text_to_ipa
from .phone_distance import (
    AVAILABLE_DISTANCE_METRICS,
    AggregateStrategy,
    PhoneDistanceCallable,
    combine_distances,
    compute_distances,
    phone_distance,
)

__all__ = [
    "text_to_ipa",
    "LanguageSegmenter",
    "AVAILABLE_DISTANCE_METRICS",
    "AggregateStrategy",
    "PhoneDistanceCallable",
    "combine_distances",
    "compute_distances",
    "phone_distance",
]
