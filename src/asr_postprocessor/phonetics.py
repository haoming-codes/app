"""Utilities for phonetic conversion and similarity scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

import epitran
import panphon.distance


@dataclass
class CharacterPhonetic:
    """Container linking a character position to its IPA representation."""

    index: int
    char: str
    ipa: str


class ChinesePhoneticizer:
    """Convert Chinese characters to IPA using Epitran.

    The phoneticizer performs a best-effort conversion. Characters that cannot
    be transliterated are skipped which prevents them from interfering with the
    phonetic distance calculation.
    """

    def __init__(self, language: str = "cmn-Hans") -> None:
        self._language = language
        self._epi = epitran.Epitran(language)

    @property
    def language(self) -> str:
        return self._language

    @lru_cache(maxsize=128)
    def transliterate(self, text: str) -> str:
        """Return the IPA representation for the supplied text."""

        return self._epi.transliterate(text)

    def ipa_tokens(self, text: str) -> List[CharacterPhonetic]:
        """Convert ``text`` into aligned IPA tokens.

        Characters that produce empty transliterations are dropped. The caller
        can therefore rely on the returned indices referencing actual
        characters from the original string.
        """

        tokens: List[CharacterPhonetic] = []
        for idx, char in enumerate(text):
            ipa = self.transliterate(char).strip()
            if not ipa:
                continue
            tokens.append(CharacterPhonetic(index=idx, char=char, ipa=ipa))
        return tokens


class PhoneticDistance:
    """Compute feature-based distances between IPA strings."""

    def __init__(self) -> None:
        self._distance = panphon.distance.Distance()

    def similarity(self, ipa_a: str, ipa_b: str) -> float:
        """Return a normalized similarity between two IPA strings.

        The distance returned by :mod:`panphon` grows with dissimilarity. We
        convert it into a similarity in the ``[0, 1]`` range where ``1`` means
        identical strings.
        """

        if not ipa_a and not ipa_b:
            return 1.0
        if not ipa_a or not ipa_b:
            return 0.0

        distance = self._distance.weighted_feature_edit_distance(ipa_a, ipa_b)
        max_len = max(len(ipa_a), len(ipa_b))
        if max_len == 0:
            return 1.0
        similarity = max(0.0, 1.0 - distance / max_len)
        return similarity

    def batch_similarity(self, ipa: str, candidates: Sequence[str]) -> List[float]:
        return [self.similarity(ipa, cand) for cand in candidates]
