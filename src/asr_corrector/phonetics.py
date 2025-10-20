"""Phonetic transcription utilities for Mandarin Chinese."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from pypinyin import Style, lazy_pinyin

_TONE_MARKS = {
    1: "˥",
    2: "˧˥",
    3: "˨˩˦",
    4: "˥˩",
    5: "˧",
}

_SPECIAL_SYLLABLES = {
    "zhi": "ʈ͡ʂɻ̩",
    "chi": "ʈ͡ʂʰɻ̩",
    "shi": "ʂɻ̩",
    "ri": "ʐɻ̩",
    "zi": "tsɨ",
    "ci": "tsʰɨ",
    "si": "sɨ",
    "yi": "i",
    "ya": "ja",
    "yao": "jau",
    "you": "jou",
    "ye": "jɛ",
    "yan": "jɛn",
    "yin": "in",
    "yang": "jɑŋ",
    "ying": "iŋ",
    "yong": "jʊŋ",
    "yue": "ɥe",
    "yuan": "ɥɛn",
    "yun": "yn",
    "yu": "y",
    "wu": "u",
    "wa": "wa",
    "wo": "wo",
    "wai": "waɪ",
    "wei": "wei",
    "wan": "wan",
    "wen": "wən",
    "wang": "wɑŋ",
    "weng": "wəŋ",
}

_INITIALS = {
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
    "zh": "ʈ͡ʂ",
    "ch": "ʈ͡ʂʰ",
    "sh": "ʂ",
    "r": "ʐ",
    "z": "ts",
    "c": "tsʰ",
    "s": "s",
    "y": "j",
    "w": "w",
    "": "",
}

_FINALS = {
    "a": "a",
    "ai": "aɪ",
    "an": "an",
    "ang": "ɑŋ",
    "ao": "au",
    "e": "ɤ",
    "ei": "ei",
    "en": "ən",
    "eng": "əŋ",
    "er": "aɻ",
    "i": "i",
    "ia": "ja",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "iao": "jau",
    "ie": "jɛ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "iu": "jou",
    "o": "uɔ",
    "ong": "ʊŋ",
    "ou": "ou",
    "u": "u",
    "ua": "wa",
    "uai": "waɪ",
    "uan": "wan",
    "uang": "wɑŋ",
    "ui": "wei",
    "uo": "wo",
    "un": "wən",
    "ueng": "wəŋ",
    "üe": "ɥe",
    "üan": "ɥɛn",
    "ün": "yn",
    "ü": "y",
}

@dataclass(frozen=True)
class SyllableIPA:
    """Stores a syllable's IPA representation with tone information."""

    ipa: str
    tone: int

    def with_tone(self) -> str:
        mark = _TONE_MARKS.get(self.tone, _TONE_MARKS[5])
        return f"{self.ipa}{mark}"


def _split_tone(syllable: str) -> tuple[str, int]:
    match = re.fullmatch(r"([a-zü:]+)([1-5])", syllable)
    if not match:
        base = syllable.replace("u:", "ü").replace("v", "ü")
        return base, 5
    base = match.group(1).replace("u:", "ü").replace("v", "ü")
    tone = int(match.group(2))
    return base, tone


def _extract_initial(base: str) -> tuple[str, str]:
    for size in (2, 1):
        initial = base[:size]
        if initial in _INITIALS and base[size:]:
            return initial, base[size:]
    if base in _SPECIAL_SYLLABLES:
        return "", base
    return "", base


def _normalise_final(initial: str, final: str) -> str:
    if initial in {"j", "q", "x"} and final.startswith("u"):
        final = "ü" + final[1:]
    return final


def _combine_initial_final(initial: str, final: str) -> str:
    ipa_initial = _INITIALS.get(initial, "")
    ipa_final = _FINALS.get(final)
    if ipa_final is None:
        raise ValueError(f"Unsupported pinyin final: {final}")
    return ipa_initial + ipa_final


def syllable_to_ipa(syllable: str) -> SyllableIPA:
    base, tone = _split_tone(syllable)
    if base in _SPECIAL_SYLLABLES:
        ipa = _SPECIAL_SYLLABLES[base]
        return SyllableIPA(ipa=ipa, tone=tone)
    initial, final = _extract_initial(base)
    final = _normalise_final(initial, final)
    ipa = _combine_initial_final(initial, final)
    return SyllableIPA(ipa=ipa, tone=tone)


def _iter_syllables(text: str) -> Iterable[str]:
    return lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        strict=False,
    )


class PhoneticTranscriber:
    """Convert Mandarin Chinese text to IPA syllable strings."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[SyllableIPA]] = {}

    def transcribe_syllables(self, text: str) -> List[SyllableIPA]:
        if text in self._cache:
            return [SyllableIPA(ipa=s.ipa, tone=s.tone) for s in self._cache[text]]
        ipa_syllables: List[SyllableIPA] = []
        for raw in _iter_syllables(text):
            if not raw:
                continue
            try:
                syllable = syllable_to_ipa(raw)
            except ValueError:
                syllable = SyllableIPA(ipa=raw, tone=5)
            ipa_syllables.append(syllable)
        self._cache[text] = [SyllableIPA(ipa=s.ipa, tone=s.tone) for s in ipa_syllables]
        return ipa_syllables

    def transcribe(self, text: str) -> str:
        syllables = self.transcribe_syllables(text)
        return " ".join(s.with_tone() for s in syllables)


__all__ = [
    "PhoneticTranscriber",
    "SyllableIPA",
    "syllable_to_ipa",
]
