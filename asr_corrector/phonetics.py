"""Phonetic utilities that bridge pypinyin and PanPhon."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

from pypinyin import Style, pinyin
from pypinyin.contrib.ipa import to_ipa


def syllables(text: str) -> Tuple[str, ...]:
    """Return tone-marked pinyin syllables for the input text."""

    if not text:
        return tuple()
    syllable_lists = pinyin(
        text,
        style=Style.TONE3,
        strict=False,
        errors=lambda item: [item],
    )
    return tuple(syllables[0] for syllables in syllable_lists)


@lru_cache(maxsize=4096)
def syllables_to_ipa(syllable_sequence: Sequence[str]) -> str:
    """Convert a syllable sequence into a space-delimited IPA string."""

    ipa_syllables = []
    for syllable in syllable_sequence:
        try:
            ipa_syllables.append(to_ipa(syllable))
        except KeyError:
            ipa_syllables.append(syllable)
    return " ".join(ipa_syllables)


@lru_cache(maxsize=4096)
def hanzi_to_ipa(text: str) -> str:
    """Convert Chinese text directly to an IPA string."""

    return syllables_to_ipa(syllables(text))
