"""Phonetic encoders used to derive IPA-like representations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BasePhoneticEncoder(ABC):
    """Interface for converting arbitrary text into a phonetic representation."""

    @abstractmethod
    def encode(self, text: str) -> str:
        """Return an IPA-like representation for *text*."""


class IdentityEncoder(BasePhoneticEncoder):
    """A trivial encoder mainly intended for testing."""

    def encode(self, text: str) -> str:
        return text


class MappingEncoder(BasePhoneticEncoder):
    """Encode characters using a mapping from graphemes to syllables."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = dict(mapping)

    def encode(self, text: str) -> str:
        return " ".join(self._mapping.get(char, char) for char in text)


class PinyinPanphonEncoder(BasePhoneticEncoder):
    """Encode Chinese text by converting to IPA via pypinyin and panphon."""

    def __init__(self, tone_sandhi: bool = True):
        try:
            from pypinyin import lazy_pinyin  # type: ignore
            from pypinyin.contrib.ipa import lazy_pinyin as lazy_pinyin_ipa  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
            raise ModuleNotFoundError(
                "PinyinPanphonEncoder requires the optional 'phonetics' extra: "
                "pip install asr-corrector[phonetics]"
            ) from exc

        try:
            from panphon.tone_models import ToneSandhi  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional sandhi handling
            ToneSandhi = None

        self._lazy_pinyin = lazy_pinyin
        self._lazy_pinyin_ipa = lazy_pinyin_ipa
        self._tone_sandhi = ToneSandhi() if (tone_sandhi and ToneSandhi is not None) else None

    def encode(self, text: str) -> str:  # pragma: no cover - depends on optional dependency
        syllables: Iterable[str] = self._lazy_pinyin(text, errors="ignore")
        if self._tone_sandhi:
            syllables = self._tone_sandhi.apply(syllables)
        ipa_syllables = self._lazy_pinyin_ipa(text, errors="ignore")
        return " ".join(ipa_syllables)


__all__ = [
    "BasePhoneticEncoder",
    "IdentityEncoder",
    "MappingEncoder",
    "PinyinPanphonEncoder",
]
