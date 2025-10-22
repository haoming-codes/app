"""Utilities for multilingual phonetic transcription."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import panphon
from phonemizer.backend import EspeakBackend
from pypinyin import Style, pinyin

_HAN_RANGE = (
    "\u3400-\u4dbf"  # Extension A
    "\u4e00-\u9fff"  # Unified ideographs
    "\uf900-\ufaff"  # Compatibility ideographs
)

_HAN_RE = re.compile(f"[{_HAN_RANGE}]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(
    rf"[{_HAN_RANGE}]|[A-Za-z]+|\d+|[^\s]", re.UNICODE
)
_DIGIT_RE = re.compile(r"[0-9]")
_STRIP_RE = re.compile(r"[\s\-\|ˈˌ\.]")

_ft = panphon.FeatureTable()
_FEATURE_DIM = len(_ft.names)


@dataclass
class PhoneticTranscription:
    """Container for the phonetic representation of a string."""

    ipa: str
    segments: List[str]
    feature_vectors: List[np.ndarray]
    tone_sequence: List[int]
    tone_unit_count: int
    stress_sequence: List[int]
    stress_unit_count: int


def _is_chinese_char(char: str) -> bool:
    return bool(_HAN_RE.fullmatch(char))


def _is_latin_word(token: str) -> bool:
    return bool(_LATIN_RE.fullmatch(token[0])) if token else False


@lru_cache(maxsize=4)
def _load_backend(language: str, with_stress: bool = False) -> EspeakBackend:
    return EspeakBackend(language=language, with_stress=with_stress)


class MultilingualPhonemizer:
    """Phonemize Chinese and English text into a shared representation."""

    def __init__(self) -> None:
        self._cn_backend = _load_backend("cmn-latn-pinyin", with_stress=False)
        self._en_backend = _load_backend("en-us", with_stress=True)

    def _phonemize_chinese(self, text: str) -> Tuple[str, List[int]]:
        ipa_parts: List[str] = []
        tones: List[int] = []
        for char in text:
            if not _is_chinese_char(char):
                continue
            raw = self._cn_backend.phonemize([char])[0]
            base = _STRIP_RE.sub("", _DIGIT_RE.sub("", raw))
            ipa_parts.append(base)
            tone_info = pinyin(char, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
            if tone_info and tone_info[0]:
                tone_match = _DIGIT_RE.search(tone_info[0][0])
                tones.append(int(tone_match.group()) if tone_match else 5)
            else:
                tones.append(5)
        return "".join(ipa_parts), tones

    def _phonemize_english_word(self, word: str) -> Tuple[str, int]:
        display = word
        if word.isupper() and len(word) > 1:
            display = " ".join(list(word))
        raw = self._en_backend.phonemize([display])[0]
        stress_level = 1 if "ˈ" in raw else 2 if "ˌ" in raw else 0
        cleaned = _STRIP_RE.sub("", raw)
        return cleaned, stress_level

    def _tokenize(self, text: str) -> List[str]:
        return [match.group(0) for match in _TOKEN_RE.finditer(text)]

    def transcribe(self, text: str) -> PhoneticTranscription:
        ipa_parts: List[str] = []
        tone_sequence: List[int] = []
        stress_sequence: List[int] = []
        chinese_units = 0
        english_units = 0

        for token in self._tokenize(text):
            if _is_chinese_char(token):
                chinese_units += 1
                ipa_piece, tones = self._phonemize_chinese(token)
                ipa_parts.append(ipa_piece)
                tone_sequence.extend(tones)
            elif _is_latin_word(token):
                english_units += 1
                ipa_piece, stress = self._phonemize_english_word(token)
                ipa_parts.append(ipa_piece)
                stress_sequence.append(stress)
            else:
                # Non-alphabetic tokens are ignored for phonetic comparison.
                continue

        ipa = "".join(ipa_parts)
        if not ipa:
            segments: List[str] = []
            vectors: List[np.ndarray] = []
        else:
            segments = _ft.segs_safe(ipa)
            vectors = [_vectorize(seg) for seg in segments]

        return PhoneticTranscription(
            ipa=ipa,
            segments=segments,
            feature_vectors=vectors,
            tone_sequence=tone_sequence,
            tone_unit_count=chinese_units,
            stress_sequence=stress_sequence,
            stress_unit_count=english_units,
        )


def _vectorize(segment: str) -> np.ndarray:
    try:
        vec = _ft.segment_to_vector(segment)
    except AttributeError:
        vec = None
    if vec is None:
        return np.zeros(_FEATURE_DIM, dtype=float)
    mapping = {"+": 1.0, "-": -1.0, "0": 0.0}
    return np.array([mapping.get(value, 0.0) for value in vec], dtype=float)


def batch_transcribe(phonemizer: MultilingualPhonemizer, texts: Sequence[str]) -> List[PhoneticTranscription]:
    """Convenience helper to transcribe a sequence of strings."""

    return [phonemizer.transcribe(text) for text in texts]
