"""Utilities for converting bilingual Chinese/English text to IPA."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import epitran

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
            return "cmn-Hans"
        if _ENGLISH_RE.match(segment):
            return "eng-Latn"
        return None


def _validate_kwargs(kwargs: dict[str, object]) -> None:
    if "language" in kwargs:
        raise ValueError("The 'language' argument is determined automatically and must not be provided.")
    kwargs.pop("backend", None)


@lru_cache(maxsize=None)
def _get_transliterator(language_code: str) -> epitran.Epitran:
    epitran.download.cedict()
    return epitran.Epitran(language_code, cedict_file="cedict_ts.u8", tones=True)


def _transliterate(language_code: str, text: str, **kwargs: object) -> str:
    transliterator = _get_transliterator(language_code)
    return transliterator.transliterate(text, **kwargs)


def text_to_ipa(
    text: str,
    *,
    segmenter: LanguageSegmenter | None = None,
    get_tone_marks: bool = False,
    **phonemize_kwargs,
) -> str:
    """Convert Chinese/English text into IPA using epitran."""
    if segmenter is None:
        segmenter = LanguageSegmenter()

    _validate_kwargs(phonemize_kwargs)

    segments = segmenter.split(text)
    grouped_segments: List[tuple[str | None, str]] = []

    current_text: str = ""
    current_language: str | None = None

    for segment in segments:
        language_code = segmenter.classify(segment)

        if not current_text:
            current_text = segment
            current_language = language_code
            continue

        if language_code == current_language or (
            language_code is None and current_language is not None
        ):
            current_text += segment
            continue

        if current_language is None and language_code is not None:
            grouped_segments.append((current_language, current_text))
            current_text = segment
            current_language = language_code
            continue

        grouped_segments.append((current_language, current_text))
        current_text = segment
        current_language = language_code

    if current_text:
        grouped_segments.append((current_language, current_text))

    result: List[str] = []
    result_tone_marks: List[str] = []

    for language_code, segment in grouped_segments:
        if language_code is None:
            result.append(segment)
            continue

        ipa = _transliterate(language_code, segment, **phonemize_kwargs)
        tone_marks = re.sub(r'\D', ' ', ipa)
        phones = re.sub(r"[ˈ\d]", "", ipa)
        # if language_code == "cmn" and remove_chinese_tone_marks:
        #     ipa = re.sub(r"\d", "", ipa)
        # if language_code.startswith("en") and remove_english_spaces:
        #     ipa = re.sub(r"\s+", "", ipa)
        result.append(phones)
        result_tone_marks.append(tone_marks)

    results = ["".join(result)]
    if get_tone_marks:
        results.append("".join(result_tone_marks))

    return results[0] if len(results) == 1 else tuple(results)


__all__ = ["text_to_ipa", "LanguageSegmenter"]
