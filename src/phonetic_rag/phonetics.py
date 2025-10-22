"""Utilities for converting multilingual text to IPA and features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple
import panphon
from phonemizer import phonemize
from pypinyin import Style, pinyin


_CJK_RANGE = re.compile(r"[\u4e00-\u9fff]")
_STRESS_PATTERN = re.compile(r"[ˈˌ]")
_DIGIT_PATTERN = re.compile(r"\d")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class PhoneticRepresentation:
    """Container with IPA, feature vectors, and suprasegmental metadata."""

    original: str
    ipa: str
    phones: List[str]
    feature_vectors: List[List[float]]
    tone_sequence: List[int]
    stress_sequence: List[str]
    chinese_char_count: int
    english_word_count: int

    def __post_init__(self) -> None:
        if len(self.phones) != len(self.feature_vectors):
            raise ValueError("phones and feature_vectors must be aligned")


def _looks_like_acronym(token: str) -> bool:
    stripped = re.sub(r"[^A-Z0-9]", "", token)
    return bool(stripped) and stripped.isupper() and len(stripped) >= 2


def _spell_out_acronym(token: str) -> str:
    mapping = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    pieces: List[str] = []
    for char in token:
        if char.isalpha():
            pieces.append(char)
        elif char.isdigit():
            pieces.append(mapping[char])
    return " ".join(pieces)


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RANGE.search(text))


def _hanzi_to_pinyin(text: str) -> Tuple[List[str], List[int]]:
    tonal_syllables = pinyin(
        text,
        style=Style.TONE3,
        errors="ignore",
        strict=False,
        neutral_tone_with_five=True,
    )
    syllables: List[str] = []
    tones: List[int] = []
    for item in tonal_syllables:
        if not item:
            continue
        syllable = item[0]
        if not syllable:
            continue
        syllables.append(syllable)
        match = re.search(r"([0-5])$", syllable)
        tones.append(int(match.group(1)) if match else 5)
    return syllables, tones


def _strip_non_segmental(text: str) -> str:
    without_digits = _DIGIT_PATTERN.sub("", text)
    without_stress = _STRESS_PATTERN.sub("", without_digits)
    return _WHITESPACE.sub("", without_stress)


def _extract_stress_sequence(ipa: str) -> List[str]:
    return [match.group(0) for match in _STRESS_PATTERN.finditer(ipa)]


def _vectors_from_phone(ft: panphon.FeatureTable, phone: str, dims: int) -> List[float]:
    vectors = ft.word_to_vector_list(phone)
    if not vectors:
        return [0.0] * dims
    symbol_map = {"+": 1.0, "-": -1.0, "0": 0.0}
    return [symbol_map.get(symbol, 0.0) for symbol in vectors[0]]


def _tokenize_runs(text: str) -> List[Tuple[str, str]]:
    """Split text into runs tagged with their coarse language."""

    runs: List[Tuple[str, str]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if _is_cjk(text[index]):
            start = index
            index += 1
            while index < len(text) and _is_cjk(text[index]):
                index += 1
            runs.append((text[start:index], "cmn"))
        elif text[index].isalpha():
            start = index
            index += 1
            while index < len(text) and text[index].isalpha():
                index += 1
            runs.append((text[start:index], "en"))
        elif text[index].isdigit():
            start = index
            index += 1
            while index < len(text) and text[index].isdigit():
                index += 1
            runs.append((text[start:index], "en"))
        else:
            runs.append((text[index], "punct"))
            index += 1
    return runs


class PhoneticTranscriber:
    """Convert multilingual strings to IPA strings and articulatory features."""

    def __init__(self, backend: str = "espeak") -> None:
        self.backend = backend
        self._feature_table = panphon.FeatureTable()
        sample_vector = self._feature_table.word_to_vector_list("a")
        self._feature_dims = len(sample_vector[0]) if sample_vector else 24

    def transcribe(
        self,
        text: str,
        *,
        treat_all_caps_as_acronyms: bool = True,
    ) -> PhoneticRepresentation:
        if not text:
            return PhoneticRepresentation(
                original=text,
                ipa="",
                phones=[],
                feature_vectors=[],
                tone_sequence=[],
                stress_sequence=[],
                chinese_char_count=0,
                english_word_count=0,
            )

        ipa_parts: List[str] = []
        tones: List[int] = []
        stress: List[str] = []
        chinese_chars = 0
        english_words = 0

        for token, lang in _tokenize_runs(text):
            if lang == "cmn":
                chinese_chars += len(token)
                syllables, token_tones = _hanzi_to_pinyin(token)
                if not syllables:
                    continue
                tones.extend(token_tones)
                pinyin_text = " ".join(syllables)
                ipa = phonemize(
                    pinyin_text,
                    language="cmn-latn-pinyin",
                    backend=self.backend,
                    strip=True,
                    preserve_punctuation=False,
                    with_stress=True,
                    language_switch="remove-flags",
                )
                ipa_parts.append(ipa)
            elif lang == "en":
                english_words += 1
                normalized = token
                if treat_all_caps_as_acronyms and _looks_like_acronym(token):
                    normalized = _spell_out_acronym(token)
                ipa = phonemize(
                    normalized,
                    language="en-us",
                    backend=self.backend,
                    strip=True,
                    preserve_punctuation=False,
                    with_stress=True,
                    language_switch="remove-flags",
                )
                stress.extend(_extract_stress_sequence(ipa))
                ipa_parts.append(ipa)
            else:
                continue

        ipa_with_marks = " ".join(ipa_parts)
        base_for_segmentation = _strip_non_segmental(ipa_with_marks)
        phones = [seg for seg in self._feature_table.ipa_segs(base_for_segmentation) if seg]
        feature_vectors = [_vectors_from_phone(self._feature_table, phone, self._feature_dims) for phone in phones]
        ipa_no_spaces = "".join(phones)

        return PhoneticRepresentation(
            original=text,
            ipa=ipa_no_spaces,
            phones=phones,
            feature_vectors=feature_vectors,
            tone_sequence=tones,
            stress_sequence=stress,
            chinese_char_count=chinese_chars,
            english_word_count=english_words,
        )

    def ipa(self, text: str, *, treat_all_caps_as_acronyms: bool = True) -> str:
        """Convenience wrapper returning only the IPA string."""
        return self.transcribe(text, treat_all_caps_as_acronyms=treat_all_caps_as_acronyms).ipa
