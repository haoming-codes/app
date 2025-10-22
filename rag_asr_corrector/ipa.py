"""IPA conversion helpers for multilingual text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from panphon import FeatureTable
from phonemizer import phonemize
from pypinyin import Style, lazy_pinyin

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]+")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z\.]+$")


@dataclass
class IPAResult:
    """Container for IPA conversion results."""

    original: str
    ipa: str
    segments: List[str]
    feature_vectors: List[np.ndarray]
    tones: List[int]
    stress_pattern: List[int]
    chinese_char_count: int
    english_word_count: int


class MultilingualIPAConverter:
    """Convert Chinese/English strings to a shared IPA representation."""

    def __init__(self, english_language: str = "en-us") -> None:
        self._ft = FeatureTable()
        self.english_language = english_language

    def ipa(self, text: str) -> IPAResult:
        ipa_parts: List[str] = []
        tones: List[int] = []
        stress_pattern: List[int] = []
        chinese_chars = 0
        english_words = 0

        for token, kind in self._tokenize(text):
            if kind == "zh":
                ipa_chunk, tone_values = self._phonemize_chinese(token)
                ipa_parts.extend(ipa_chunk)
                tones.extend(tone_values)
                chinese_chars += len(tone_values)
            elif kind == "en":
                ipa_word, stress = self._phonemize_english(token)
                ipa_parts.append(ipa_word)
                stress_pattern.append(stress)
                english_words += 1
            else:
                continue

        ipa_string = "".join(ipa_parts)
        segments = self._ft.ipa_segs(ipa_string)
        feature_vectors = [self._vectorize(seg) for seg in segments]

        return IPAResult(
            original=text,
            ipa=ipa_string,
            segments=segments,
            feature_vectors=feature_vectors,
            tones=tones,
            stress_pattern=stress_pattern,
            chinese_char_count=chinese_chars,
            english_word_count=english_words,
        )

    def ipa_segments(self, text: str) -> Sequence[str]:
        return self.ipa(text).segments

    def ipa_string(self, text: str) -> str:
        return self.ipa(text).ipa

    def _phonemize_chinese(self, token: str) -> Tuple[List[str], List[int]]:
        syllables = lazy_pinyin(
            token,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        tone_values: List[int] = []
        processed: List[str] = []
        numbered_syllables: List[str] = []
        for syll in syllables:
            match = re.match(r"([a-zA-Z]+)([0-5])?", syll)
            if not match:
                continue
            base, tone = match.group(1), match.group(2)
            tone_values.append(int(tone) if tone else 5)
            numbered_syllables.append(f"{base}{tone_values[-1]}")
        if not numbered_syllables:
            return [], []
        ipa_chunk = phonemize(
            " ".join(numbered_syllables),
            language="cmn-latn-pinyin",
            backend="espeak",
            strip=True,
            language_switch="remove-flags",
        )
        processed = ipa_chunk.split()
        return processed, tone_values

    def _phonemize_english(self, token: str) -> Tuple[str, int]:
        text = token
        if _ACRONYM_RE.match(token):
            letters = re.sub(r"[^A-Z]", "", token)
            text = " ".join(letters)
        ipa_word = phonemize(
            text,
            language=self.english_language,
            backend="espeak",
            strip=True,
            language_switch="remove-flags",
            with_stress=True,
        )
        stress = self._extract_stress_level(ipa_word)
        ipa_clean = ipa_word.replace("ˈ", "").replace("ˌ", "")
        ipa_clean = ipa_clean.replace(" ", "")
        return ipa_clean, stress

    @staticmethod
    def _extract_stress_level(ipa_word: str) -> int:
        level = 0
        if "ˈ" in ipa_word:
            level = 2
        elif "ˌ" in ipa_word:
            level = 1
        return level

    def _vectorize(self, segment: str) -> np.ndarray:
        vector = self._ft.segment_to_vector(segment)
        mapping = {"+": 1.0, "-": -1.0, "0": 0.0}
        return np.array([mapping[val] for val in vector], dtype=float)

    def _tokenize(self, text: str) -> Iterable[Tuple[str, str]]:
        i = 0
        while i < len(text):
            char = text[i]
            if _CHINESE_RE.match(char):
                j = i + 1
                while j < len(text) and _CHINESE_RE.match(text[j]):
                    j += 1
                yield text[i:j], "zh"
                i = j
            elif char.isalpha():
                j = i + 1
                while j < len(text) and text[j].isalpha():
                    j += 1
                yield text[i:j], "en"
                i = j
            else:
                i += 1
