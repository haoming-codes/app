"""Utilities for converting multilingual text to IPA and feature representations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from panphon import FeatureTable
from phonemizer.backend import EspeakBackend
import regex as reg


# Characters that indicate suprasegmental information in espeak IPA output
_PRIMARY_STRESS = "ˈ"
_SECONDARY_STRESS = "ˌ"
_STRESS_MARKS = {_PRIMARY_STRESS, _SECONDARY_STRESS}
_TONE_PATTERN = re.compile(r"[0-9]")


def _symbol_to_float(symbol: str) -> float:
    if symbol == '+':
        return 1.0
    if symbol == '-':
        return -1.0
    return 0.0


@dataclass
class IPAResult:
    """Representation of IPA transcription and derived features."""

    ipa: str
    phones: List[str]
    feature_vectors: np.ndarray
    tones: List[int]
    stresses: List[int]
    syllables: List[str]

    def __post_init__(self) -> None:
        if isinstance(self.feature_vectors, list):
            self.feature_vectors = np.asarray(self.feature_vectors, dtype=float)


class IPAConverter:
    """Convert multilingual text to IPA, articulatory features, and suprasegmentals."""

    _ZH_RANGE = reg.compile(r"[\p{IsHan}]+")
    _LATIN_RANGE = reg.compile(r"[\p{Latin}0-9]+")

    def __init__(self, language_hint: Optional[str] = None) -> None:
        self.language_hint = language_hint
        self._feature_table = FeatureTable()
        self._backends: dict[str, EspeakBackend] = {}

    def _backend(self, language: str) -> EspeakBackend:
        if language not in self._backends:
            self._backends[language] = EspeakBackend(
                language=language,
                punctuation_marks=';:,.!?¡¿—…“”()[]',
                with_stress=True,
                language_switch='remove-flags',
            )
        return self._backends[language]

    def _normalize_for_acronym(self, text: str) -> str:
        letters = [letter for letter in text if letter.isalpha()]
        return " ".join(letters)

    def segment_text(self, text: str, language_hint: Optional[str] = None) -> List[tuple[str, str]]:
        """Segment text into homogeneous language chunks."""

        if language_hint:
            return [(text, language_hint)]
        if self.language_hint:
            return [(text, self.language_hint)]

        chunks: List[tuple[str, str]] = []
        idx = 0
        while idx < len(text):
            if text[idx].isspace() or reg.match(r"\p{P}+", text[idx]):
                idx += 1
                continue
            if self._ZH_RANGE.match(text, idx):
                match = self._ZH_RANGE.match(text, idx)
                assert match is not None
                chunks.append((match.group(0), 'cmn'))
                idx = match.end()
                continue
            if self._LATIN_RANGE.match(text, idx):
                match = self._LATIN_RANGE.match(text, idx)
                assert match is not None
                chunks.append((match.group(0), 'en-us'))
                idx = match.end()
                continue
            # default to English backend
            chunks.append((text[idx], 'en-us'))
            idx += 1
        return chunks

    def _detect_chunks(self, text: str) -> List[tuple[str, str]]:
        return self.segment_text(text)

    def phonemize_chunk(self, text: str, language: str) -> str:
        backend = self._backend(language)
        return backend.phonemize([text], strip=True)[0]

    def to_ipa(
        self,
        text: str,
        *,
        is_acronym: bool = False,
        language_hint: Optional[str] = None,
    ) -> IPAResult:
        """Convert text to IPA and derive features.

        Parameters
        ----------
        text:
            Input multilingual string.
        is_acronym:
            Treat the text as an acronym pronounced letter-by-letter.
        language_hint:
            Override automatic language detection with a backend key.
        """

        if is_acronym:
            text = self._normalize_for_acronym(text)

        if language_hint:
            chunks = [(text, language_hint)]
        else:
            chunks = self._detect_chunks(text)

        ipa_parts: List[str] = []
        syllables: List[str] = []
        for chunk_text, lang in chunks:
            ipa_chunk = self.phonemize_chunk(chunk_text, lang)
            ipa_parts.append(ipa_chunk)
            syllables.extend([s for s in ipa_chunk.split() if s])
        ipa_combined = " ".join(part for part in ipa_parts if part)

        tones, stresses = self._extract_suprasegmentals(syllables)
        cleaned = self._strip_suprasegmentals(ipa_combined)
        phones = self._feature_table.ipa_segs(cleaned)
        vectors = []
        for phone in phones:
            vector = self._feature_table.segment_to_vector(phone)
            if vector is None:
                vector = [0.0] * self._feature_table.dim
            vectors.append([_symbol_to_float(v) for v in vector])
        return IPAResult(
            ipa=ipa_combined.replace(" ", ""),
            phones=phones,
            feature_vectors=np.asarray(vectors, dtype=float),
            tones=tones,
            stresses=stresses,
            syllables=syllables,
        )

    @staticmethod
    def _extract_suprasegmentals(syllables: Sequence[str]) -> tuple[List[int], List[int]]:
        tones: List[int] = []
        stresses: List[int] = []
        for syll in syllables:
            digits = [int(d) for d in _TONE_PATTERN.findall(syll)]
            tones.extend(digits)
            if _PRIMARY_STRESS in syll:
                stresses.append(2)
            elif _SECONDARY_STRESS in syll:
                stresses.append(1)
            else:
                stresses.append(0)
        return tones, stresses

    @staticmethod
    def _strip_suprasegmentals(ipa: str) -> str:
        cleaned_chars: List[str] = []
        for ch in ipa:
            if ch.isspace():
                continue
            if ch in _STRESS_MARKS or ch.isdigit() or ch in "()":
                continue
            cleaned_chars.append(ch)
        return "".join(cleaned_chars)
