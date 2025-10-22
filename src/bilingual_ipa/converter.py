"""Utilities for converting bilingual Chinese/English text to IPA."""
from __future__ import annotations

import re
from typing import List

from phonemizer import phonemize

_CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_ENGLISH_RE = re.compile(r"^[A-Za-z]+$")
_SEGMENT_RE = re.compile(r"([\u4e00-\u9fff]+|[A-Za-z]+|\s+|[^\u4e00-\u9fffA-Za-z\s]+)")


class LanguageSegmenter:
    """Splits bilingual text into language-aware chunks."""

    def split(self, text: str) -> List[str]:
        if not text:
            return []
        return _SEGMENT_RE.findall(text)

    def classify(self, segment: str) -> str | None:
        if _CHINESE_RE.match(segment):
            return "cmn"
        if _ENGLISH_RE.match(segment):
            return "en"
        return None


def _validate_kwargs(kwargs: dict[str, object]) -> None:
    if "language" in kwargs:
        raise ValueError("The 'language' argument is determined automatically and must not be provided.")
    backend = kwargs.get("backend")
    if backend is not None and backend != "espeak":
        raise ValueError("Only the 'espeak' backend is supported.")
    kwargs.pop("backend", None)


def text_to_ipa(text: str, *, segmenter: LanguageSegmenter | None = None, **phonemize_kwargs) -> str:
    """Convert Chinese/English text into IPA using phonemizer."""
    if segmenter is None:
        segmenter = LanguageSegmenter()

    _validate_kwargs(phonemize_kwargs)

    segments = segmenter.split(text)
    result: List[str] = []

    for segment in segments:
        language_code = segmenter.classify(segment)
        if language_code is None:
            result.append(segment)
            continue
        ipa = phonemize(segment, language=language_code, backend="espeak", **phonemize_kwargs)
        result.append(ipa)

    return "".join(result)


__all__ = ["text_to_ipa", "LanguageSegmenter"]
