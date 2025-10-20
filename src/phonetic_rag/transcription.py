"""Text to IPA transcription utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from eng_to_ipa import convert as english_to_ipa
from pypinyin import Style, lazy_pinyin

_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class TranscribedSequence:
    """Container for IPA transcription and tones."""

    ipa: str
    de_toned_ipa: str
    tones: Tuple[int, ...]


def _contains_chinese(text: str) -> bool:
    return bool(_CHINESE_CHAR_RE.search(text))


_INITIALS = [
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
]

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
    "r": "ɻ",
    "z": "ts",
    "c": "tsʰ",
    "s": "s",
}

_SPECIAL_SYLLABLES = {
    "zhi": "ʈʂɻ̩",
    "chi": "ʈʂʰɻ̩",
    "shi": "ʂɻ̩",
    "ri": "ɻ̩",
    "zi": "tsz̩",
    "ci": "tsʰz̩",
    "si": "sz̩",
}

_FINAL_TO_IPA = {
    "a": "a",
    "ai": "ai̯",
    "an": "an",
    "ang": "ɑŋ",
    "ao": "au̯",
    "e": "ɤ",
    "ei": "ei̯",
    "en": "ən",
    "eng": "əŋ",
    "er": "ɑɻ",
    "ia": "ja",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "iao": "jau̯",
    "ie": "jɛ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "iu": "jou̯",
    "i": "i",
    "iong": "jʊŋ",
    "o": "uɔ",
    "ong": "ʊŋ",
    "ou": "ou̯",
    "ua": "wa",
    "uai": "wai̯",
    "uan": "wan",
    "uang": "wɑŋ",
    "ui": "wei̯",
    "uo": "uɔ",
    "un": "wən",
    "u": "u",
    "üe": "yɛ",
    "üan": "yɛn",
    "ün": "yn",
    "ü": "y",
    "ve": "yɛ",
    "van": "yɛn",
    "vn": "yn",
    "uei": "wei̯",
    "uen": "wən",
}


def _normalize_syllable(syllable: str) -> Tuple[str, int]:
    tone = 5
    base = syllable
    if base and base[-1].isdigit():
        tone = int(base[-1])
        base = base[:-1]
    return base, tone


def _split_initial_final(base: str) -> Tuple[str, str]:
    for initial in sorted(_INITIALS, key=len, reverse=True):
        if base.startswith(initial):
            return initial, base[len(initial) :]
    return "", base


def _handle_zero_initial(base: str) -> Tuple[str, str]:
    if base.startswith("y"):
        if base.startswith("yu"):
            return "", "ü" + base[2:]
        if base.startswith("yi"):
            return "", "i" + base[2:]
        return "", "i" + base[1:]
    if base.startswith("w"):
        return "", "u" + base[1:]
    return "", base


def _syllable_to_ipa(syllable: str) -> Tuple[str, int]:
    base, tone = _normalize_syllable(syllable)
    if not base:
        return "", tone
    if base in _SPECIAL_SYLLABLES:
        return _SPECIAL_SYLLABLES[base], tone
    initial, final = _split_initial_final(base)
    if not initial:
        initial, final = _handle_zero_initial(base)
    ipa_initial = _INITIAL_TO_IPA.get(initial, "")
    final_key = final.replace("u:", "ü").replace("v", "ü")
    ipa_final = _FINAL_TO_IPA.get(final_key, final_key)
    ipa = (ipa_initial + ipa_final).strip()
    return ipa, tone


class Transcriber:
    """Transcribe multilingual text into IPA sequences and tone lists."""

    def __init__(self) -> None:
        pass

    @lru_cache(maxsize=2048)
    def transcribe(self, text: str) -> TranscribedSequence:
        if not text:
            return TranscribedSequence("", "", tuple())
        if _contains_chinese(text):
            ipa_tokens: List[str] = []
            tones: List[int] = []
            syllables = lazy_pinyin(
                text,
                style=Style.TONE3,
                neutral_tone_with_five=True,
                strict=False,
            )
            for syllable in syllables:
                ipa, tone = _syllable_to_ipa(syllable)
                if ipa:
                    ipa_tokens.append(ipa)
                tones.append(tone)
            ipa = " ".join(ipa_tokens)
            return TranscribedSequence(ipa, ipa, tuple(tones))
        ipa = english_to_ipa(text)
        ipa = ipa.replace("ˈ", "").replace("ˌ", "")
        ipa = re.sub(r"\s+", " ", ipa).strip()
        return TranscribedSequence(ipa, ipa, tuple())
