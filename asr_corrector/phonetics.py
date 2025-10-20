"""Phonetic utilities for Mandarin Chinese."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from pypinyin import Style, pinyin


INITIAL_TO_IPA = {
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


FINAL_TO_IPA = {
    "a": "a",
    "o": "ɔ",
    "e": "ɤ",
    "ê": "ɛ",
    "i": "i",
    "u": "u",
    "v": "y",
    "ai": "aɪ",
    "ei": "eɪ",
    "ao": "ɑʊ",
    "ou": "oʊ",
    "an": "an",
    "en": "ən",
    "ang": "aŋ",
    "eng": "əŋ",
    "ong": "ʊŋ",
    "ia": "ja",
    "io": "jɔ",
    "ie": "jɛ",
    "iu": "jou",
    "ian": "jɛn",
    "in": "in",
    "iao": "jɑʊ",
    "iang": "jɑŋ",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "ua": "wa",
    "uo": "wo",
    "uai": "waɪ",
    "ui": "wei",
    "uan": "wan",
    "uang": "wɑŋ",
    "un": "wən",
    "ueng": "wəŋ",
    "üe": "yɛ",
    "üan": "yæn",
    "ün": "yn",
    "er": "ɑɻ",
}


@dataclass
class PhoneticSequence:
    """A phonetic transcription broken into syllables."""

    segments: List[str]
    tones: List[int]

    def __post_init__(self) -> None:  # pragma: no cover - defensive programming
        if len(self.segments) != len(self.tones):
            raise ValueError("Segments and tones must have the same length")

    def as_ipa(self) -> str:
        """Return the concatenated IPA string for segmental scoring."""

        return " ".join(self.segments)


def _strip_tone(final: str) -> tuple[str, int]:
    tone = 5
    base = final
    for digit in "12345":
        if digit in base:
            tone = int(digit)
            base = base.replace(digit, "")
    return base, tone


def _normalise_final(base: str, initial: str) -> str:
    if base == "" and initial in {"z", "c", "s"}:
        return "ɿ"
    if base == "" and initial in {"zh", "ch", "sh", "r"}:
        return "ʅ"
    if base == "i" and initial in {"z", "c", "s"}:
        return "ɿ"
    if base == "i" and initial in {"zh", "ch", "sh", "r"}:
        return "ʅ"
    if base == "u" and initial in {"j", "q", "x", "y"}:
        base = "v"
    if base == "ue" and initial in {"j", "q", "x", "y"}:
        base = "üe"
    if base == "uan" and initial in {"j", "q", "x", "y"}:
        base = "üan"
    if base == "un" and initial in {"j", "q", "x", "y"}:
        base = "ün"
    return base


def _final_to_ipa(base: str, initial: str) -> str:
    base = _normalise_final(base, initial)
    return FINAL_TO_IPA.get(base, base)


def _initial_to_ipa(initial: str) -> str:
    return INITIAL_TO_IPA.get(initial, initial)


def to_phonetic_sequence(text: str) -> PhoneticSequence:
    """Convert Chinese text to IPA segments and tone numbers."""

    if not text:
        return PhoneticSequence([], [])

    initials = pinyin(text, style=Style.INITIALS, strict=False)
    finals = pinyin(text, style=Style.FINALS_TONE3, strict=False)

    segments: List[str] = []
    tones: List[int] = []
    for initial_list, final_list in zip(initials, finals):
        initial = initial_list[0] if initial_list else ""
        final = final_list[0] if final_list else ""
        base_final, tone = _strip_tone(final)
        tone = tone if tone != 0 else 5
        ipa_initial = _initial_to_ipa(initial)
        ipa_final = _final_to_ipa(base_final, initial)
        segment = (ipa_initial + ipa_final).strip()
        segments.append(segment if segment else ipa_final)
        tones.append(tone)
    return PhoneticSequence(segments, tones)


def batch_to_phonetic_sequence(tokens: Iterable[str]) -> List[PhoneticSequence]:
    """Convenience helper to convert many tokens."""

    return [to_phonetic_sequence(token) for token in tokens]
