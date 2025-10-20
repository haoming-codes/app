"""Utilities for converting Chinese text into phonetic representations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from pypinyin import Style, lazy_pinyin


_TONE_RE = re.compile(r"^([a-züv:]+?)([1-5]?)$")
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Syllable:
    """Represents the phonetic information for a single Hanzi character."""

    char: str
    pinyin: Optional[str]
    tone: Optional[int]
    ipa: Optional[str]
    valid: bool = True


class PinyinConversionError(ValueError):
    """Raised when a syllable cannot be converted to IPA."""


class PinyinConverter:
    """Converts Chinese text to :class:`Syllable` objects with IPA transcriptions."""

    def __init__(self) -> None:
        self._ipa_converter = _PinyinToIpa()

    def text_to_syllables(self, text: str) -> List[Syllable]:
        """Convert *text* to a list of :class:`Syllable` objects.

        Non-Chinese characters are marked as invalid syllables to simplify
        downstream filtering logic.
        """

        pinyin_list = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        syllables: List[Syllable] = []
        for char, pinyin in zip(text, pinyin_list):
            if not _CHINESE_CHAR_RE.fullmatch(char):
                syllables.append(Syllable(char=char, pinyin=None, tone=None, ipa=None, valid=False))
                continue

            base, tone = _split_pinyin_tone(pinyin)
            ipa = self._ipa_converter.to_ipa(base)
            syllables.append(Syllable(char=char, pinyin=base, tone=tone, ipa=ipa, valid=True))
        return syllables


class _PinyinToIpa:
    """Converts tone-stripped Hanyu Pinyin syllables to IPA strings.

    The converter implements a rule-based mapping from pinyin initials and
    finals to IPA. It focuses on accuracy for Mandarin Chinese and intentionally
    omits tone information, as tones are handled separately.
    """

    _SPECIAL_SYLLABLES = {
        "zhi": "ʈʂɻ̩",
        "chi": "ʈʂʰɻ̩",
        "shi": "ʂɻ̩",
        "ri": "ʐɻ̩",
        "zi": "tsɨ",
        "ci": "tsʰɨ",
        "si": "sɨ",
        "zhiu": "ʈʂjou̯",  # rare but allows graceful handling
        "zii": "tsɨ",
        "cih": "tsʰɨ",
    }

    _INITIALS = (
        "zh",
        "ch",
        "sh",
        "b",
        "p",
        "m",
        "f",
        "d",
        "t",
        "n",
        "l",
        "g",
        "k",
        "h",
        "j",
        "q",
        "x",
        "r",
        "z",
        "c",
        "s",
        "y",
        "w",
    )

    _INITIAL_IPA = {
        "b": "p",
        "p": "pʰ",
        "m": "m",
        "f": "f",
        "d": "t",
        "t": "tʰ",
        "n": "n",
        "l": "l",
        "g": "k",
        "k": "kʰ",
        "h": "x",
        "j": "tɕ",
        "q": "tɕʰ",
        "x": "ɕ",
        "zh": "ʈʂ",
        "ch": "ʈʂʰ",
        "sh": "ʂ",
        "r": "ʐ",
        "z": "ts",
        "c": "tsʰ",
        "s": "s",
        "y": "j",
        "w": "w",
    }

    _FINAL_IPA = {
        "a": "a",
        "o": "ɔ",
        "e": "ɤ",
        "ai": "ai̯",
        "ei": "ei̯",
        "ao": "au̯",
        "ou": "ou̯",
        "an": "an",
        "en": "ən",
        "ang": "ɑŋ",
        "eng": "əŋ",
        "ong": "ʊŋ",
        "i": "i",
        "ia": "ia",
        "iao": "iau̯",
        "ian": "iɛn",
        "iang": "iɑŋ",
        "ie": "iɛ",
        "in": "in",
        "ing": "iŋ",
        "iong": "jʊŋ",
        "iu": "jou̯",
        "ua": "ua",
        "uai": "uai̯",
        "uan": "uan",
        "uang": "uɑŋ",
        "ui": "uei̯",
        "uo": "uo",
        "un": "uən",
        "u": "u",
        "üe": "yɛ",
        "üan": "yɛn",
        "ün": "yn",
        "ü": "y",
        "er": "aɻ",
    }

    _ZERO_INITIAL_ALTERNATIVES = {
        "a": "a",
        "ai": "ai̯",
        "an": "an",
        "ang": "ɑŋ",
        "ao": "au̯",
        "e": "ɤ",
        "ei": "ei̯",
        "en": "ən",
        "eng": "əŋ",
        "er": "aɻ",
        "o": "uɔ",
        "ong": "ʊŋ",
        "ou": "ou̯",
        "i": "i",
        "ia": "ja",
        "ian": "jɛn",
        "iang": "jɑŋ",
        "iao": "jau̯",
        "ie": "jɛ",
        "in": "in",
        "ing": "iŋ",
        "iong": "jʊŋ",
        "iu": "jou̯",
        "u": "u",
        "ua": "wa",
        "uai": "wai̯",
        "uan": "wan",
        "uang": "wɑŋ",
        "ui": "wei̯",
        "uo": "wo",
        "un": "wən",
        "üe": "yɛ",
        "üan": "yɛn",
        "ün": "yn",
        "ü": "y",
    }

    def to_ipa(self, pinyin: str) -> str:
        pinyin = pinyin.lower().replace("u:", "ü").replace("v", "ü")
        if pinyin in self._SPECIAL_SYLLABLES:
            return self._SPECIAL_SYLLABLES[pinyin]

        initial, final = self._split_initial_final(pinyin)

        if not final:
            raise PinyinConversionError(f"Cannot parse pinyin syllable: {pinyin}")

        if initial == "y":
            initial = ""
            if final.startswith("u"):
                final = "ü" + final[1:]
            ipa = self._ZERO_INITIAL_ALTERNATIVES.get(final)
            if ipa is None:
                raise PinyinConversionError(f"Unsupported y-initial syllable: {pinyin}")
            return ipa

        if initial == "w":
            initial = ""
            ipa = self._ZERO_INITIAL_ALTERNATIVES.get(final)
            if ipa is None:
                raise PinyinConversionError(f"Unsupported w-initial syllable: {pinyin}")
            return ipa

        initial_ipa = self._INITIAL_IPA.get(initial, "")
        if not initial and final in self._ZERO_INITIAL_ALTERNATIVES:
            return self._ZERO_INITIAL_ALTERNATIVES[final]

        final_ipa = self._FINAL_IPA.get(final)
        if final_ipa is None:
            raise PinyinConversionError(f"Unsupported pinyin final: {final} from {pinyin}")

        return f"{initial_ipa}{final_ipa}"

    def _split_initial_final(self, pinyin: str) -> tuple[str, str]:
        for initial in self._INITIALS:
            if pinyin.startswith(initial):
                return initial, pinyin[len(initial) :]
        return "", pinyin


def _split_pinyin_tone(pinyin_with_tone: str) -> tuple[str, int]:
    match = _TONE_RE.match(pinyin_with_tone.lower())
    if not match:
        raise PinyinConversionError(f"Invalid pinyin syllable: {pinyin_with_tone}")
    base, tone_str = match.groups()
    tone = int(tone_str) if tone_str else 5
    return base, tone


def is_chinese_char(char: str) -> bool:
    return bool(_CHINESE_CHAR_RE.fullmatch(char))


def syllables_for_terms(converter: PinyinConverter, terms: Iterable[str]) -> List[List[Syllable]]:
    """Pre-compute syllable lists for canonical terms."""

    results: List[List[Syllable]] = []
    for term in terms:
        syllables = converter.text_to_syllables(term)
        if not all(s.valid for s in syllables):
            raise PinyinConversionError(f"Term '{term}' contains characters that cannot be converted to pinyin")
        results.append(syllables)
    return results
