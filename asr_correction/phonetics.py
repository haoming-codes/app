from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, List, Sequence, Tuple

from pypinyin import Style, pinyin

try:
    from eng_to_ipa import convert as english_to_ipa
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The 'eng_to_ipa' package is required for English transliteration. "
        "Install it with `pip install eng_to_ipa`."
    ) from exc


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

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
    "y",
    "w",
]

_INITIAL_IPA = {
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

_FINAL_IPA = {
    "a": "a",
    "ai": "aɪ",
    "an": "an",
    "ang": "ɑŋ",
    "ao": "ɑʊ",
    "e": "ɤ",
    "ei": "eɪ",
    "en": "ən",
    "eng": "əŋ",
    "er": "aɻ",
    "i": "i",
    "ia": "ja",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "iao": "jɑʊ",
    "ie": "jɛ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "iu": "joʊ",
    "o": "o",
    "ong": "ʊŋ",
    "ou": "oʊ",
    "u": "u",
    "ua": "wa",
    "uai": "waɪ",
    "uan": "wan",
    "uang": "wɑŋ",
    "ui": "weɪ",
    "un": "wən",
    "uo": "wo",
    "ü": "y",
    "üe": "ɥe",
    "ün": "yn",
    "iaó": "jɑʊ",  # fallback for mis-normalised syllables
}

_SPECIAL_FINALS = {
    "zhi": "ʈʂʅ",
    "chi": "ʈʂʰʅ",
    "shi": "ʂʅ",
    "ri": "ʐʅ",
    "zi": "tsɿ",
    "ci": "tsʰɿ",
    "si": "sɿ",
}

_TONE_PATTERN = re.compile(r"([a-z:ü]+)([1-5]?)")


@dataclass
class MandarinSyllable:
    ipa: str
    tone: int


@dataclass
class PhoneticRepresentation:
    ipa: str
    tones: Tuple[int, ...]

    @property
    def detoned(self) -> str:
        return self.ipa


@lru_cache(maxsize=1024)
def _split_initial_final(syllable: str) -> Tuple[str, str]:
    for initial in _INITIALS:
        if syllable.startswith(initial) and initial != "":
            return initial, syllable[len(initial) :]
    return "", syllable


def _normalise_pinyin(syllable: str) -> Tuple[str, int]:
    match = _TONE_PATTERN.fullmatch(syllable)
    if not match:
        return syllable, 5
    base, tone = match.groups()
    base = base.replace("u:", "ü").replace("v", "ü")
    tone_val = int(tone) if tone else 5
    return base, tone_val


def mandarin_syllable_to_ipa(syllable: str) -> MandarinSyllable:
    syllable = syllable.lower()
    if syllable in _SPECIAL_FINALS:
        tone = 5
        m = _TONE_PATTERN.fullmatch(syllable)
        if m and m.group(2):
            tone = int(m.group(2))
        base = m.group(1) if m else syllable
        return MandarinSyllable(_SPECIAL_FINALS[base], tone)

    base, tone = _normalise_pinyin(syllable)

    if base in _SPECIAL_FINALS:
        return MandarinSyllable(_SPECIAL_FINALS[base], tone)

    initial, final = _split_initial_final(base)

    # Handle syllables that start with y/w which act as vowels.
    if initial == "y":
        if final.startswith("u") or final.startswith("ü"):
            initial = ""
        else:
            initial = "j"
    if initial == "w":
        initial = "w"

    # Resolve ü cases with j/q/x.
    if initial in {"j", "q", "x"} and final.startswith("u"):
        final = "ü" + final[1:]

    final_ipa = _FINAL_IPA.get(final)
    if final_ipa is None:
        # Attempt to strip leading vowel markers introduced by y/w handling.
        for key in sorted(_FINAL_IPA.keys(), key=len, reverse=True):
            if final.endswith(key):
                final_ipa = _FINAL_IPA[key]
                break
    if final_ipa is None:
        final_ipa = final

    initial_ipa = _INITIAL_IPA.get(initial, "")
    ipa = f"{initial_ipa}{final_ipa}" if final_ipa else initial_ipa
    return MandarinSyllable(ipa, tone)


def mandarin_text_to_representation(text: str) -> PhoneticRepresentation:
    syllables: List[MandarinSyllable] = []
    raw = pinyin(text, style=Style.TONE3, strict=False, errors=lambda chars: [c for c in chars])
    for candidates in raw:
        if not candidates:
            continue
        candidate = candidates[0]
        if not _TONE_PATTERN.fullmatch(candidate):
            continue
        syllables.append(mandarin_syllable_to_ipa(candidate))
    ipa = " ".join(s.ipa for s in syllables)
    tones = tuple(s.tone for s in syllables)
    return PhoneticRepresentation(ipa, tones)


def english_text_to_representation(text: str) -> PhoneticRepresentation:
    ipa = english_to_ipa(text)
    tones: Tuple[int, ...] = tuple()
    return PhoneticRepresentation(ipa, tones)


def text_to_representation(text: str, language: str) -> PhoneticRepresentation:
    language = language.lower()
    if language in {"zh", "cmn", "mandarin"}:
        return mandarin_text_to_representation(text)
    if language in {"en", "eng", "english"}:
        return english_text_to_representation(text)
    raise ValueError(f"Unsupported language '{language}'.")
