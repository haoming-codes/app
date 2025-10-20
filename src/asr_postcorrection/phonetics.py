"""Utilities for converting Mandarin Chinese text into approximate IPA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pypinyin import Style
from pypinyin.core import Pinyin


# Tone marks approximated for Mandarin tones. The neutral tone is represented as a mid-level tone.
_TONE_MARKS = {
    "1": "˥",
    "2": "˧˥",
    "3": "˨˩˦",
    "4": "˥˩",
    "5": "˧",
}

# Mapping from Pinyin initials to IPA approximations.
_INITIALS_TO_IPA = {
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

# Mapping from Pinyin finals (without tone numbers) to IPA approximations.
# The mapping intentionally favours transparency over extreme phonetic accuracy.
_FINALS_TO_IPA = {
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
    "ua": "wa",
    "uai": "wai",
    "uo": "wo",
    "ui": "wei",
    "uan": "wan",
    "un": "wən",
    "uang": "wɑŋ",
    "ueng": "wəŋ",
    "u": "u",
    "üe": "yɛ",
    "ü": "y",
    "ün": "yn",
    "er": "ɑɻ",
}


def _apply_tone(ipa: str, tone: str) -> str:
    mark = _TONE_MARKS.get(tone, _TONE_MARKS["5"])
    return f"{ipa}{mark}" if ipa else mark


def _resolve_special_i(initial: str, final: str) -> str:
    """Handle special cases for the high central vowel written as ``i`` in Pinyin."""
    if final != "i":
        return _FINALS_TO_IPA.get(final, final)
    if initial in {"zh", "ch", "sh", "r"}:
        return "ɻ̩"
    if initial in {"z", "c", "s"}:
        return "ɿ"
    return "i"


def _normalize_final(initial: str, final: str) -> str:
    base_final = final
    if base_final in {"iu", "ui"}:
        base_final = {"iu": "iou", "ui": "uei"}[base_final]
    if initial in {"j", "q", "x", "y"} and base_final.startswith("u"):
        base_final = base_final.replace("u", "ü", 1)
    if base_final == "ong" and initial in {"j", "q", "x"}:
        base_final = "iong"
    if base_final == "i" and initial == "y":
        return "i"
    if base_final == "u" and initial == "w":
        return "u"
    return base_final


def _split_tone(final_with_tone: str) -> tuple[str, str]:
    tone = "5"
    base = final_with_tone
    if final_with_tone and final_with_tone[-1].isdigit():
        tone = final_with_tone[-1]
        base = final_with_tone[:-1]
    return base, tone


def _convert_syllable(initial: str, final_with_tone: str) -> str:
    base_final, tone = _split_tone(final_with_tone)
    base_final = _normalize_final(initial, base_final)
    if base_final == "" and initial == "":
        return ""
    if base_final == "i":
        ipa_final = _resolve_special_i(initial, base_final)
    else:
        ipa_final = _FINALS_TO_IPA.get(base_final, base_final)
    ipa_initial = _INITIALS_TO_IPA.get(initial, initial)
    return _apply_tone(f"{ipa_initial}{ipa_final}", tone)


def text_to_ipa_segments(text: str, *, pinyin: Pinyin | None = None) -> List[str]:
    """Convert Mandarin text to a list of IPA syllable approximations.

    Parameters
    ----------
    text:
        Input text consisting primarily of Chinese characters.
    pinyin:
        Optional :class:`pypinyin.core.Pinyin` instance. Reusing an instance
        avoids repeatedly loading the model dictionary when processing many
        phrases.
    """

    if not text:
        return []

    engine = pinyin or Pinyin()
    initials = engine.pinyin(text, style=Style.INITIALS, strict=False)
    finals = engine.pinyin(
        text,
        style=Style.FINALS_TONE3,
        strict=False,
        neutral_tone_with_five=True,
    )
    segments: List[str] = []
    for initial_list, final_list in zip(initials, finals):
        initial = initial_list[0] if initial_list else ""
        final = final_list[0] if final_list else ""
        segments.append(_convert_syllable(initial, final))
    return [segment for segment in segments if segment]


def text_to_ipa(text: str, *, pinyin: Pinyin | None = None) -> str:
    """Convert text to a whitespace-delimited IPA string."""
    return " ".join(text_to_ipa_segments(text, pinyin=pinyin))


@dataclass(frozen=True)
class SyllableIPA:
    """Container storing a syllable and its IPA approximation."""

    hanzi: str
    ipa: str


def text_to_syllable_ipa(text: str) -> List[SyllableIPA]:
    """Return IPA approximations paired with each Chinese character."""
    segments = text_to_ipa_segments(text)
    return [SyllableIPA(hanzi=char, ipa=seg) for char, seg in zip(text, segments)]


__all__ = [
    "SyllableIPA",
    "text_to_ipa",
    "text_to_ipa_segments",
    "text_to_syllable_ipa",
]
