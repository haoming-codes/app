"""Phonetic similarity helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence


class PhoneticDistance:
    """Compute similarity between IPA sequences.

    The implementation prefers :mod:`panphon` when available. When the
    dependency is missing the class falls back to a simple string similarity
    over the IPA sequences. This keeps the correction pipeline usable in
    constrained environments while still favouring phonetic signals when the
    richer dependency set is installed.
    """

    def __init__(self) -> None:
        try:
            from panphon.distance import Distance  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._distance = None
        else:
            self._distance = Distance()

    def similarity(self, ipa_a: Sequence[str], ipa_b: Sequence[str]) -> float:
        """Return a similarity score in the range ``[0, 1]``."""

        joined_a = " ".join(ipa_a)
        joined_b = " ".join(ipa_b)

        if self._distance is not None:
            # ``weighted_feature_edit_distance`` is roughly proportional to the
            # number of substitutions needed. We normalise it by the longer
            # sequence to obtain a bounded similarity score.
            distance = self._distance.weighted_feature_edit_distance(joined_a, joined_b)
            normaliser = max(len(ipa_a), len(ipa_b), 1)
            similarity = 1.0 - min(distance / normaliser, 1.0)
            return similarity

        return SequenceMatcher(None, joined_a, joined_b).ratio()


__all__ = ["PhoneticDistance"]
