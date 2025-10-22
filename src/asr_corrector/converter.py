"""Multilingual text to IPA conversion utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence
from panphon.featuretable import FeatureTable
from pypinyin import Style, pinyin
from phonemizer import phonemize

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
_FEATURE_VALUE = {"+": 1, "-": -1, "0": 0}


@dataclass(frozen=True)
class TokenPhonetics:
    """Phonetic data for a single token."""

    token: str
    language: str
    ipa: str
    tone_sequence: Sequence[int]
    stress_level: int | None


@dataclass(frozen=True)
class PhoneticSequence:
    """Aggregate phonetic representation of a string."""

    ipa: str
    features: List[List[int]]
    tone_sequence: List[int]
    stress_sequence: List[int]
    tokens: List[TokenPhonetics]


class MultilingualPhoneticConverter:
    """Convert Chinese/English strings into IPA and articulatory features."""

    def __init__(
        self,
        english_language: str = "en-us",
        chinese_language: str = "cmn-latn-pinyin",
        tone_neutral: int = 5,
    ) -> None:
        self.english_language = english_language
        self.chinese_language = chinese_language
        self.tone_neutral = tone_neutral
        self._feature_table = FeatureTable()

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z\.]+", text)]

    @staticmethod
    def _is_chinese(token: str) -> bool:
        return bool(_CHINESE_RE.fullmatch(token))

    @staticmethod
    def _is_acronym(token: str) -> bool:
        letters = [ch for ch in token if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters)

    def _pinyin_tokens(self, token: str) -> List[str]:
        syllables = [item[0] for item in pinyin(token, style=Style.TONE3, strict=False)]
        return syllables

    @lru_cache(maxsize=1024)
    def _phonemize(self, text: str, language: str) -> str:
        return phonemize(
            text,
            language=language,
            backend="espeak",
            strip=True,
            with_stress=True,
            language_switch="remove-flags",
        )

    def _ipa_for_chinese(self, token: str) -> TokenPhonetics:
        syllables = self._pinyin_tokens(token)
        joined = " ".join(syllables)
        ipa = self._phonemize(joined, self.chinese_language)
        tones = [self._extract_tone(syl) for syl in syllables]
        return TokenPhonetics(token, "zh", ipa, tones, None)

    def _ipa_for_acronym(self, token: str) -> TokenPhonetics:
        letters = [ch for ch in token if ch.isalpha()]
        text = " ".join(letters)
        ipa = self._phonemize(text, self.english_language)
        stress = 1 if "ˈ" in ipa else 0
        return TokenPhonetics(token, "en", ipa, (), stress)

    def _ipa_for_english(self, token: str) -> TokenPhonetics:
        ipa = self._phonemize(token, self.english_language)
        stress = 2 if "ˌ" in ipa else (1 if "ˈ" in ipa else 0)
        return TokenPhonetics(token, "en", ipa, (), stress)

    def to_sequence(self, text: str) -> PhoneticSequence:
        tokens = []
        ipa_parts: List[str] = []
        features: List[List[int]] = []
        tone_sequence: List[int] = []
        stress_sequence: List[int] = []

        for token in self._tokenize(text):
            if self._is_chinese(token):
                token_phonetics = self._ipa_for_chinese(token)
                tone_sequence.extend(token_phonetics.tone_sequence)
            elif self._is_acronym(token):
                token_phonetics = self._ipa_for_acronym(token)
                stress_sequence.append(token_phonetics.stress_level or 0)
            else:
                token_phonetics = self._ipa_for_english(token)
                stress_sequence.append(token_phonetics.stress_level or 0)

            tokens.append(token_phonetics)
            ipa_parts.append(token_phonetics.ipa)
            features.extend(self._features_for(token_phonetics.ipa))

        combined_ipa = " ".join(ipa_parts)
        return PhoneticSequence(combined_ipa, features, tone_sequence, stress_sequence, tokens)

    def _features_for(self, ipa: str) -> List[List[int]]:
        feature_vectors = self._feature_table.word_to_vector_list(ipa)
        numeric_vectors: List[List[int]] = []
        for vector in feature_vectors:
            numeric_vectors.append([_FEATURE_VALUE.get(value, 0) for value in vector])
        return numeric_vectors

    def _extract_tone(self, syllable: str) -> int:
        match = re.search(r"(\d)", syllable)
        if match:
            return int(match.group(1))
        return self.tone_neutral

    def ipa(self, text: str) -> str:
        """Return IPA transcription for *text*."""

        return self.to_sequence(text).ipa
