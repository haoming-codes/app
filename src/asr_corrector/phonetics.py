"""Utilities for converting Chinese text to phonetic representations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from pypinyin import Style, pinyin

_TONE_RE = re.compile(r"^([a-zA-Z:üÜv]+)([1-5]?)$")

# Mapping from Pinyin initials to IPA. Values are approximations that work well for
# measuring similarity. The pseudo-initials ``y`` and ``w`` are handled specially to
# account for glide insertion rules.
_INITIAL_IPA: Dict[str, str] = {
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
}

# Special syllable-to-IPA mappings that cannot be derived through a regular
# initial/final decomposition.
_SPECIAL_SYLLABLES: Dict[str, str] = {
    "zhi": "ʈʂɻ̩",
    "chi": "ʈʂʰɻ̩",
    "shi": "ʂɻ̩",
    "ri": "ʐɻ̩",
    "zi": "tsɿ",
    "ci": "tsʰɿ",
    "si": "sɿ",
    "er": "aɻ",
    "ng": "ŋ",
    "n": "n",
    "m": "m",
}

# Mapping of Pinyin finals (after normalisation) to IPA. The values favour
# readability over extreme phonetic precision; their role is to provide inputs to
# distance metrics.
_FINAL_IPA: Dict[str, str] = {
    "a": "a",
    "ai": "aɪ̯",
    "an": "an",
    "ang": "ɑŋ",
    "ao": "ɑʊ̯",
    "e": "ɤ",
    "ei": "eɪ̯",
    "en": "ən",
    "eng": "əŋ",
    "er": "aɻ",
    "o": "o",
    "ou": "oʊ̯",
    "ong": "ʊŋ",
    "i": "i",
    "ia": "ia",
    "ian": "iɛn",
    "iang": "iɑŋ",
    "iao": "iɑʊ̯",
    "ie": "iɛ",
    "in": "in",
    "ing": "iŋ",
    "iong": "iʊŋ",
    "iu": "ioʊ̯",
    "u": "u",
    "ua": "ua",
    "uai": "uaɪ̯",
    "uan": "uan",
    "uang": "uɑŋ",
    "ui": "ueɪ̯",
    "uo": "uo",
    "un": "uən",
    "ü": "y",
    "üe": "ɥe",
    "üan": "ɥɛn",
    "ün": "yn",
}


@dataclass
class Syllable:
    """Container storing a single syllable with its phonetic metadata."""

    char: str
    pinyin: str
    base: str
    tone: int
    ipa: str


@dataclass
class PhoneticSequence:
    """Collection of syllables describing a full string."""

    text: str
    syllables: Sequence[Syllable]

    @property
    def ipa_syllables(self) -> List[str]:
        return [s.ipa for s in self.syllables if s.ipa]

    @property
    def tones(self) -> List[int]:
        return [s.tone for s in self.syllables if s.base]

    @property
    def phonetic_length(self) -> int:
        return max(len([s for s in self.syllables if s.base]), 1)

    def concatenate(self, separator: str = "") -> str:
        return separator.join(self.ipa_syllables)


class PhoneticConverter:
    """Convert Chinese strings into IPA syllable sequences."""

    def __init__(self) -> None:
        self._cache: Dict[str, PhoneticSequence] = {}

    def convert(self, text: str) -> PhoneticSequence:
        if text in self._cache:
            return self._cache[text]

        py_syllables = pinyin(text, style=Style.TONE3, heteronym=False, strict=False)
        syllables: List[Syllable] = []
        for char, syllable_list in zip(text, py_syllables):
            if not syllable_list:
                syllables.append(Syllable(char=char, pinyin="", base="", tone=5, ipa=""))
                continue
            entry = syllable_list[0]
            base, tone = self._split_tone(entry)
            if not base:
                syllables.append(Syllable(char=char, pinyin=entry, base="", tone=tone, ipa=""))
                continue
            try:
                ipa = self._pinyin_to_ipa(base)
            except ValueError:
                ipa = base
            syllables.append(Syllable(char=char, pinyin=entry, base=base, tone=tone, ipa=ipa))

        sequence = PhoneticSequence(text=text, syllables=tuple(syllables))
        self._cache[text] = sequence
        return sequence

    @staticmethod
    def _split_tone(pinyin_syllable: str) -> Tuple[str, int]:
        match = _TONE_RE.match(pinyin_syllable.lower())
        if not match:
            return pinyin_syllable.lower(), 5
        base = match.group(1).replace("v", "ü").replace("u:", "ü")
        tone = int(match.group(2) or 5)
        return base, tone

    def _pinyin_to_ipa(self, syllable: str) -> str:
        syllable = syllable.lower()
        if syllable in _SPECIAL_SYLLABLES:
            return _SPECIAL_SYLLABLES[syllable]

        initial, final = self._split_initial_final(syllable)
        final = self._normalize_final(initial, final)
        initial_ipa = self._initial_to_ipa(initial, final)
        final_ipa = self._final_to_ipa(initial, final)
        if initial_ipa and final_ipa.startswith(initial_ipa):
            final_ipa = final_ipa[len(initial_ipa) :]
        return f"{initial_ipa}{final_ipa}"

    @staticmethod
    def _split_initial_final(syllable: str) -> Tuple[str, str]:
        for initial in ("zh", "ch", "sh"):
            if syllable.startswith(initial):
                return initial, syllable[len(initial) :]
        for initial in (
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
        ):
            if syllable.startswith(initial):
                return initial, syllable[len(initial) :]
        return "", syllable

    @staticmethod
    def _normalize_final(initial: str, final: str) -> str:
        final = final.replace("v", "ü").replace("u:", "ü")
        if not final:
            return final
        if final == "o" and initial not in {"", "w", "y"}:
            final = "uo"
        if final == "ong" and initial == "y":
            return "iong"
        if initial in {"j", "q", "x"} or (initial == "y" and final.startswith("u")):
            if final.startswith("u"):
                final = "ü" + final[1:]
        if initial == "y" and final.startswith("i") and final not in {"i", "in", "ing"}:
            final = final[1:]
        if initial == "w" and final.startswith("u") and final not in {"u"}:
            final = final[1:]
        return final

    def _initial_to_ipa(self, initial: str, final: str) -> str:
        if initial == "":
            return ""
        if initial == "y":
            if final in {"i", "in", "ing", "ü", "ün"}:
                return ""
            if final.startswith("ü"):
                return "ɥ"
            return "j"
        if initial == "w":
            if final == "u":
                return ""
            return "w"
        return _INITIAL_IPA.get(initial, "")

    def _final_to_ipa(self, initial: str, final: str) -> str:
        if final in _SPECIAL_SYLLABLES:
            return _SPECIAL_SYLLABLES[final]
        if final not in _FINAL_IPA:
            raise ValueError(f"Unsupported Pinyin final: {final}")
        ipa = _FINAL_IPA[final]
        return ipa
