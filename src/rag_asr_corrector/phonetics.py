from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

from panphon.featuretable import FeatureTable
from phonemizer.backend import EspeakBackend

from .config import DistanceConfig


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_ENGLISH_RE = re.compile(r"[A-Za-z0-9\.']")
_SANITIZE_RE = re.compile(r"[0-9ˈˌ\s\(\)\[\]\{\}\-]")
_ACRONYM_RE = re.compile(r"^[A-Z0-9]+(?:\.[A-Z0-9]+)*$")


@dataclass
class BilingualToken:
    """Token produced by the bilingual tokenizer."""

    text: str
    language: str  # "cmn" or "en"
    start: int
    end: int


@dataclass
class PronouncedToken:
    """Pronunciation information for a single token."""

    token: BilingualToken
    ipa: str
    sanitized: str
    tone: Optional[int] = None
    stress: Optional[str] = None


@dataclass
class Pronunciation:
    """Aggregated pronunciation for a string."""

    tokens: List[PronouncedToken]
    sanitized: str
    feature_vectors: List[List[int]]
    chinese_tones: List[int]
    english_stress: List[str]

    def ipa_tokens(self) -> List[str]:
        return [token.ipa for token in self.tokens]


class PhonemizerService:
    """Wraps phonemizer backends for Chinese and English."""

    def __init__(self, config: DistanceConfig) -> None:
        self.config = config
        self._feature_table = FeatureTable()

    @staticmethod
    def _is_chinese_char(char: str) -> bool:
        return bool(_CHINESE_RE.match(char))

    def tokenize(self, text: str) -> List[BilingualToken]:
        tokens: List[BilingualToken] = []
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            if self._is_chinese_char(ch):
                tokens.append(BilingualToken(ch, "cmn", i, i + 1))
                i += 1
                continue
            if ch.isspace():
                i += 1
                continue
            if _ENGLISH_RE.match(ch):
                j = i
                while j < length and _ENGLISH_RE.match(text[j]):
                    j += 1
                tokens.append(BilingualToken(text[i:j], "en", i, j))
                i = j
                continue
            # Skip punctuation characters that are not part of tokens.
            i += 1
        return tokens

    def phonemize(self, text: str) -> Pronunciation:
        tokens = self.tokenize(text)
        pronounced_tokens: List[PronouncedToken] = []
        tones: List[int] = []
        stresses: List[str] = []
        sanitized_parts: List[str] = []
        for token in tokens:
            ipa = self._phonemize_token(token)
            sanitized = _sanitize_ipa(ipa)
            tone = None
            stress = None
            if token.language == "cmn":
                tone = _extract_tone(ipa)
                if tone is not None:
                    tones.append(tone)
            elif token.language == "en":
                stress = _extract_stress(ipa)
                if stress is not None:
                    stresses.append(stress)
            pronounced_tokens.append(
                PronouncedToken(
                    token=token,
                    ipa=ipa,
                    sanitized=sanitized,
                    tone=tone,
                    stress=stress,
                )
            )
            sanitized_parts.append(sanitized)
        sanitized_full = "".join(sanitized_parts)
        feature_vectors = self._feature_table.word_to_vector_list(
            sanitized_full, numeric=True
        )
        return Pronunciation(
            tokens=pronounced_tokens,
            sanitized=sanitized_full,
            feature_vectors=feature_vectors,
            chinese_tones=tones,
            english_stress=stresses,
        )

    def ipa(self, text: str) -> List[PronouncedToken]:
        return self.phonemize(text).tokens

    def _phonemize_token(self, token: BilingualToken) -> str:
        if token.language == "cmn":
            backend = self._get_backend(self.config.phonemizer_language_cmn)
            ipa = backend.phonemize([token.text])[0]
            return ipa.strip()
        backend = self._get_backend(self.config.phonemizer_language_en)
        prepared_text = _prepare_english_token(token.text)
        ipa = backend.phonemize([prepared_text])[0]
        return ipa.strip()

    @lru_cache(maxsize=None)
    def _get_backend(self, language: str) -> EspeakBackend:
        return EspeakBackend(
            language=language,
            preserve_punctuation=False,
            with_stress=True,
            language_switch="remove-flags",
        )

    @property
    def feature_table(self) -> FeatureTable:
        return self._feature_table


def _prepare_english_token(token: str) -> str:
    stripped = token.strip()
    if _ACRONYM_RE.match(stripped):
        letters = re.sub(r"[^A-Z0-9]", "", stripped)
        return " ".join(list(letters))
    return stripped


def _sanitize_ipa(ipa: str) -> str:
    cleaned = ipa
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace("`", "")
    cleaned = _SANITIZE_RE.sub("", cleaned)
    return cleaned


def _extract_tone(ipa: str) -> Optional[int]:
    match = re.search(r"([1-5])", ipa)
    if match:
        return int(match.group(1))
    return None


def _extract_stress(ipa: str) -> Optional[str]:
    if "ˈ" in ipa:
        return "ˈ"
    if "ˌ" in ipa:
        return "ˌ"
    return "0"


def prepare_pronunciation(text: str, config: DistanceConfig) -> Pronunciation:
    service = PhonemizerService(config)
    return service.phonemize(text)


def ipa_tokens(text: str, config: Optional[DistanceConfig] = None) -> List[PronouncedToken]:
    if config is None:
        config = DistanceConfig()
    service = PhonemizerService(config)
    return service.ipa(text)
