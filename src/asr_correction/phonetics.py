"""Utilities for converting multilingual text into IPA and articulatory features."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from panphon import FeatureTable
from phonemizer.backend import EspeakBackend
from pypinyin import Style, pinyin


_STRESS_MARKERS = {"ˈ": "primary", "ˌ": "secondary"}
_CJK_BLOCKS = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)

_MANDARIN_INITIAL_IPA = {
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

_MANDARIN_FINAL_IPA = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ê": "ɛ",
    "ai": "aɪ",
    "ei": "eɪ",
    "ao": "ɑʊ",
    "ou": "oʊ",
    "an": "an",
    "en": "ən",
    "ang": "ɑŋ",
    "eng": "əŋ",
    "er": "ɚ",
    "ong": "ʊŋ",
    "i": "i",
    "ia": "ja",
    "iao": "jɑʊ",
    "ie": "jɛ",
    "iu": "joʊ",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "io": "jo",
    "ua": "wa",
    "uo": "wo",
    "uai": "waɪ",
    "ui": "weɪ",
    "uan": "wan",
    "uang": "wɑŋ",
    "un": "wən",
    "ueng": "wəŋ",
    "u": "u",
    "ue": "we",
    "uen": "wən",
    "ü": "y",
    "üe": "yɛ",
    "üan": "ɥɛn",
    "ün": "yn",
    "v": "y",
    "ve": "yɛ",
    "van": "ɥɛn",
    "vn": "yn",
}

_SPECIAL_SYLLABLES = {
    ("zh", "i"): "ʈʂɻ̩",
    ("ch", "i"): "ʈʂʰɻ̩",
    ("sh", "i"): "ʂɻ̩",
    ("r", "i"): "ʐɻ̩",
    ("z", "i"): "tsɿ",
    ("c", "i"): "tsʰɿ",
    ("s", "i"): "sɿ",
}

_MANDARIN_NEUTRAL_TONE = 5


@dataclass(slots=True)
class Phone:
    """Representation of a single phone with articulatory metadata."""

    symbol: str
    features: np.ndarray
    tone: Optional[int] = None
    stress: Optional[str] = None
    is_vowel: bool = False
    language: Optional[str] = None


@dataclass(slots=True)
class PhoneticSequence:
    """Container holding the IPA phones for an input string."""

    phones: List[Phone]
    ipa: str

    def tones(self) -> List[int]:
        """Return tone annotations for the syllabic nuclei."""

        return [p.tone for p in self.phones if p.tone is not None]

    def stresses(self) -> List[str]:
        """Return stress annotations for the syllabic nuclei."""

        return [p.stress for p in self.phones if p.stress]


class IPAConverter:
    """Convert multilingual strings into IPA phones and feature vectors."""

    def __init__(self, english_language: str = "en-us") -> None:
        self._feature_table = FeatureTable()
        self._syl_index = self._feature_table.names.index("syl")
        self._english_backend = EspeakBackend(
            english_language,
            with_stress=True,
            language_switch="remove-flags",
        )

    def sequence(self, text: str) -> PhoneticSequence:
        """Return a :class:`PhoneticSequence` describing ``text``."""

        phones: List[Phone] = []
        ipa_parts: List[str] = []
        for chunk, lang in self._segment_by_language(text):
            if not chunk:
                continue
            if lang == "zh":
                seq = self._mandarin_sequence(chunk)
            else:
                seq = self._english_sequence(chunk)
            phones.extend(seq.phones)
            if seq.ipa:
                ipa_parts.append(seq.ipa)
        ipa = " ".join(part for part in ipa_parts if part)
        return PhoneticSequence(phones=phones, ipa=ipa)

    def ipa(self, text: str) -> str:
        """Return the concatenated IPA transcription for ``text``."""

        return self.sequence(text).ipa

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_cjk(char: str) -> bool:
        code = ord(char)
        for start, end in _CJK_BLOCKS:
            if start <= code <= end:
                return True
        return False

    def _segment_by_language(self, text: str) -> Iterable[Tuple[str, str]]:
        buffer: List[str] = []
        current_lang: Optional[str] = None
        for char in text:
            if char.isspace():
                if buffer:
                    yield ("".join(buffer), current_lang or "en")
                    buffer = []
                    current_lang = None
                continue
            lang = "zh" if self._is_cjk(char) else "en"
            if current_lang is None:
                buffer = [char]
                current_lang = lang
            elif lang == current_lang:
                buffer.append(char)
            else:
                if buffer:
                    yield ("".join(buffer), current_lang)
                buffer = [char]
                current_lang = lang
        if buffer:
            yield ("".join(buffer), current_lang or "en")

    def _english_sequence(self, text: str) -> PhoneticSequence:
        ipa = self._english_backend.phonemize([text], strip=True)[0]
        segments = self._feature_table.ipa_segs(ipa)
        vectors = self._feature_table.word_to_vector_list(ipa)
        stress_map = self._extract_stress(ipa, segments)
        phones: List[Phone] = []
        for seg, vec, stress in zip(segments, vectors, stress_map):
            features = _vector_to_array(vec)
            is_vowel = vec[self._syl_index] == "+"
            stress_value = stress if is_vowel else None
            phones.append(
                Phone(
                    symbol=seg,
                    features=features,
                    tone=None,
                    stress=stress_value,
                    is_vowel=is_vowel,
                    language="en",
                )
            )
        return PhoneticSequence(phones=phones, ipa=ipa)

    def _mandarin_sequence(self, text: str) -> PhoneticSequence:
        phones: List[Phone] = []
        ipa_parts: List[str] = []
        initials = pinyin(
            text,
            style=Style.INITIALS,
            strict=False,
            neutral_tone_with_five=True,
        )
        finals = pinyin(
            text,
            style=Style.FINALS_TONE3,
            strict=False,
            neutral_tone_with_five=True,
        )
        for init_list, final_list in zip(initials, finals):
            initial = (init_list[0] or "").lower()
            final = (final_list[0] or "").lower()
            ipa, tone = _mandarin_to_ipa(initial, final)
            ipa_parts.append(ipa)
            syllable_phones = self._phones_from_ipa(ipa, tone, language="zh")
            phones.extend(syllable_phones)
        return PhoneticSequence(phones=phones, ipa=" ".join(ipa_parts))

    def _phones_from_ipa(
        self,
        ipa: str,
        tone: Optional[int],
        language: str,
    ) -> List[Phone]:
        segments = self._feature_table.ipa_segs(ipa)
        vectors = self._feature_table.word_to_vector_list(ipa)
        phones: List[Phone] = []
        tone_applied = False
        for seg, vec in zip(segments, vectors):
            features = _vector_to_array(vec)
            is_vowel = vec[self._syl_index] == "+"
            tone_value = None
            if is_vowel and tone and not tone_applied:
                tone_value = tone
                tone_applied = True
            phones.append(
                Phone(
                    symbol=seg,
                    features=features,
                    tone=tone_value,
                    stress=None,
                    is_vowel=is_vowel,
                    language=language,
                )
            )
        return phones

    @staticmethod
    def _extract_stress(ipa: str, segments: Sequence[str]) -> List[Optional[str]]:
        result: List[Optional[str]] = [None] * len(segments)
        pending: Optional[str] = None
        seg_idx = 0
        cursor = 0
        while seg_idx < len(segments) and cursor < len(ipa):
            char = ipa[cursor]
            if char in _STRESS_MARKERS:
                pending = _STRESS_MARKERS[char]
                cursor += 1
                continue
            if char.isspace():
                cursor += 1
                continue
            segment = segments[seg_idx]
            if ipa.startswith(segment, cursor):
                if pending:
                    result[seg_idx] = pending
                    pending = None
                cursor += len(segment)
                seg_idx += 1
            else:
                cursor += 1
        return result


def _vector_to_array(vector: Sequence[str]) -> np.ndarray:
    mapping = {"+": 1.0, "-": -1.0, "0": 0.0}
    return np.array([mapping.get(value, math.nan) for value in vector], dtype=float)


def _mandarin_to_ipa(initial: str, final_with_tone: str) -> Tuple[str, Optional[int]]:
    if not final_with_tone:
        return _MANDARIN_INITIAL_IPA.get(initial, ""), None
    tone: Optional[int] = None
    if final_with_tone[-1].isdigit():
        tone = int(final_with_tone[-1])
        if tone == 0:
            tone = _MANDARIN_NEUTRAL_TONE
        final = final_with_tone[:-1]
    else:
        final = final_with_tone
    final = final.replace("v", "v")
    if (initial, final) in _SPECIAL_SYLLABLES:
        ipa = _SPECIAL_SYLLABLES[(initial, final)]
        return ipa, tone
    base_initial = initial
    base_final = final
    if base_final.startswith("u") and initial in {"j", "q", "x", "y"}:
        base_final = "v" + base_final[1:]
    if initial == "y":
        if base_final == "i":
            base_initial = ""
        elif base_final.startswith("i") and base_final != "i":
            base_final = base_final[1:]
    if initial == "w" and base_final.startswith("u"):
        if base_final == "u":
            base_initial = ""
        else:
            base_final = base_final[1:]
    ipa_initial = _MANDARIN_INITIAL_IPA.get(base_initial, "")
    ipa_final = _MANDARIN_FINAL_IPA.get(base_final)
    if ipa_final is None:
        ipa_final = _MANDARIN_FINAL_IPA.get(final)
    if ipa_final is None:
        ipa_final = final
    ipa = ipa_initial + ipa_final
    return ipa, tone


__all__ = ["IPAConverter", "Phone", "PhoneticSequence"]
