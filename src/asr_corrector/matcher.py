"""Similarity scoring utilities."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

try:  # pragma: no cover - optional dependency
    from panphon.distance import Distance as PanphonDistance
except ImportError:  # pragma: no cover
    PanphonDistance = None


class PhoneticMatcher:
    """Score similarity between IPA strings."""

    def __init__(self) -> None:
        self._panphon: Optional[PanphonDistance] = PanphonDistance() if PanphonDistance else None

    def similarity(self, ipa_a: str, ipa_b: str) -> float:
        if not ipa_a or not ipa_b:
            return 0.0
        if self._panphon is not None:
            distance = self._panphon.weighted_feature_edit_distance(ipa_a, ipa_b)
            max_len = max(len(ipa_a), len(ipa_b), 1)
            score = 1.0 - (distance / max_len)
            return max(0.0, min(1.0, score))
        matcher = SequenceMatcher(a=ipa_a, b=ipa_b)
        return matcher.ratio()


__all__ = ["PhoneticMatcher"]
