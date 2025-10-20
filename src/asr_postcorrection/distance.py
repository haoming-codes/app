"""Phonetic distance utilities based on panphon."""
from __future__ import annotations

from functools import lru_cache

from panphon.distance import Distance


@lru_cache(maxsize=1)
def _distance() -> Distance:
    return Distance()


def similarity(ipa_a: str, ipa_b: str) -> float:
    """Returns a similarity score in [0, 1] based on feature edit distance."""
    if ipa_a == ipa_b:
        return 1.0
    if not ipa_a or not ipa_b:
        return 0.0

    dist = _distance().weighted_feature_edit_distance(ipa_a, ipa_b)
    max_len = max(len(ipa_a), len(ipa_b))
    if max_len == 0:
        return 1.0
    normalized = dist / max_len
    normalized = min(max(normalized, 0.0), 1.5)  # guardrail for extremely large distances
    return max(0.0, 1.0 - normalized)
