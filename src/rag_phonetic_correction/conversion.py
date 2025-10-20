"""Utilities for converting Chinese text into phonetic representations."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from pypinyin import Style, lazy_pinyin


_INITIALS = (
    "zh",
    "ch",
    "sh",
    "z",
    "c",
    "s",
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
    "y",
    "w",
)

_INITIAL_TO_IPA = {
    "": "",
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

_FINAL_TO_IPA = {
    "a": "a",
    "o": "uɔ",
    "e": "ɤ",
    "ai": "ai",
    "ei": "ei",
    "ao": "au",
    "ou": "ou",
    "an": "an",
    "en": "ən",
    "ang": "ɑŋ",
    "eng": "əŋ",
    "ong": "ʊŋ",
    "er": "ɑɻ",
    "i": "i",
    "ia": "ja",
    "iao": "jau",
    "ie": "jɛ",
    "iu": "jou",
    "ian": "jɛn",
    "in": "in",
    "iang": "jɑŋ",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "io": "jɔ",
    "ua": "wa",
    "uo": "uɔ",
    "uai": "wai",
    "ui": "wei",
    "uan": "wan",
    "un": "wən",
    "uang": "wɑŋ",
    "ueng": "wəŋ",
    "u": "u",
    "ue": "ɥe",
    "üe": "ɥe",
    "üan": "ɥɛn",
    "uan": "wan",
    "ün": "yn",
    "ün": "yn",
    "ü": "y",
    "ia": "ja",
    "iong": "jʊŋ",
    "iao": "jau",
    "iong": "jʊŋ",
    "iang": "jɑŋ",
    "iong": "jʊŋ",
}

# Remove duplicated keys caused by dictionary literal above by rebuilding explicitly.
_FINAL_TO_IPA = {
    **{
        "a": "a",
        "o": "uɔ",
        "e": "ɤ",
        "ai": "ai",
        "ei": "ei",
        "ao": "au",
        "ou": "ou",
        "an": "an",
        "en": "ən",
        "ang": "ɑŋ",
        "eng": "əŋ",
        "ong": "ʊŋ",
        "er": "ɑɻ",
        "i": "i",
        "ia": "ja",
        "iao": "jau",
        "ie": "jɛ",
        "iu": "jou",
        "ian": "jɛn",
        "in": "in",
        "iang": "jɑŋ",
        "ing": "iŋ",
        "iong": "jʊŋ",
        "io": "jɔ",
        "ua": "wa",
        "uo": "uɔ",
        "uai": "wai",
        "ui": "wei",
        "uan": "wan",
        "un": "wən",
        "uang": "wɑŋ",
        "ueng": "wəŋ",
        "u": "u",
        "ue": "ɥe",
        "üe": "ɥe",
        "üan": "ɥɛn",
        "ün": "yn",
        "ü": "y",
    }
}

_SPECIAL_SYLLABLES = {
    "zhi": "ʈʂɻ̩",
    "chi": "ʈʂʰɻ̩",
    "shi": "ʂɻ̩",
    "ri": "ʐɻ̩",
    "zi": "tsɿ",
    "ci": "tsʰɿ",
    "si": "sɿ",
}

_TONE_PATTERN = re.compile(r"([a-züv]+)([1-5]?)$", re.IGNORECASE)


@dataclass(frozen=True)
class PhoneticRepresentation:
    """Stores syllable-level phonetic information for a text span."""

    syllables: List[str]
    ipa: List[str]
    tones: List[int]

    def toneless_ipa(self) -> List[str]:
        """Return IPA syllables without tone marks."""
        return self.ipa


class PhoneticConverter:
    """Convert Chinese text into phonetic representations."""

    def __init__(self) -> None:
        self._initials = sorted(_INITIALS, key=len, reverse=True)

    def convert(self, text: str) -> PhoneticRepresentation:
        syllables_with_tone = lazy_pinyin(
            text,
            style=Style.TONE3,
            errors=lambda chars: list(chars),
        )
        syllables: List[str] = []
        ipa: List[str] = []
        tones: List[int] = []
        for syllable in syllables_with_tone:
            if not syllable:
                continue
            tone = self._extract_tone(syllable)
            base = self._toneless(syllable)
            syllables.append(base)
            ipa.append(self._pinyin_to_ipa(base))
            tones.append(tone)
        return PhoneticRepresentation(syllables=syllables, ipa=ipa, tones=tones)

    @staticmethod
    def _extract_tone(syllable: str) -> int:
        match = _TONE_PATTERN.match(syllable)
        if match and match.group(2):
            return int(match.group(2))
        return 5

    @staticmethod
    def _toneless(syllable: str) -> str:
        base = re.sub(r"[1-5]", "", syllable)
        return base.replace("v", "ü")

    def _pinyin_to_ipa(self, syllable: str) -> str:
        syllable_lower = syllable.lower()
        if syllable_lower in _SPECIAL_SYLLABLES:
            return _SPECIAL_SYLLABLES[syllable_lower]
        initial, final = self._split_initial_final(syllable_lower)
        final = self._adjust_final(initial, final)
        initial_ipa = _INITIAL_TO_IPA.get(initial, "")
        final_ipa = _FINAL_TO_IPA.get(final)
        if final_ipa is None:
            # Fallback: return the syllable itself if mapping is missing.
            final_ipa = final
        return f"{initial_ipa}{final_ipa}" if initial_ipa else final_ipa

    def _split_initial_final(self, syllable: str) -> tuple[str, str]:
        for initial in self._initials:
            if syllable.startswith(initial):
                return initial, syllable[len(initial) :]
        return "", syllable

    def _adjust_final(self, initial: str, final: str) -> str:
        if not final:
            return final
        if initial in {"j", "q", "x"} and final.startswith("u"):
            final = "ü" + final[1:]
        if initial == "y" and final.startswith("u"):
            final = "ü" + final[1:]
        if initial == "y" and final == "u":
            final = "ü"
        if initial == "y" and final.startswith("un"):
            final = "ün" + final[2:]
        if final == "iu":
            return "iu"
        if final == "ui":
            return "ui"
        if final == "un":
            return "un"
        return final
