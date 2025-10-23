"""Utilities for converting bilingual Chinese/English text to IPA."""
from __future__ import annotations

import re
from typing import List

from dragonmapper.hanzi import to_ipa as hanzi_to_ipa
from eng_to_ipa import convert as english_to_ipa
from dragonmapper.transcriptions import _IPA_TONES

_IPA_TONES = "".join(_IPA_TONES.values())
_STRESS = "ˈˌ"
_NON_IPA_TONES_RE = re.compile(f"[^{_IPA_TONES}]")
_NON_STRESS_RE = re.compile(f"[^{_STRESS}]")
_IPA_TONES_STRESS_RE = re.compile(f"[{_IPA_TONES}{_STRESS}]")
_CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_ENGLISH_RE = re.compile(r"^[A-Za-z]+$")
_SEGMENT_RE = re.compile(r"([\u4e00-\u9fff]+|[A-Za-z]+|\s+|[^\u4e00-\u9fffA-Za-z\s]+)")
_ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
_PUNCT_NEEDS_SPACE_RE = re.compile(r"([.,;:!?])(?!\s|$)")


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
            return "en-us"
        return None


def _normalize_english_segment(segment: str) -> str:
    """Insert spaces after punctuation and between letters in all-caps words."""

    segment = _PUNCT_NEEDS_SPACE_RE.sub(r"\1 ", segment)
    return _ALL_CAPS_WORD_RE.sub(lambda match: " ".join(match.group(0)), segment)


def text_to_ipa(
    text: str,
    *,
    segmenter: LanguageSegmenter | None = None,
    get_tone_marks: bool = False,
    get_stress_marks: bool = False,
) -> str:
    """Convert Chinese/English text into IPA using ``eng_to_ipa`` and ``dragonmapper``."""
    if segmenter is None:
        segmenter = LanguageSegmenter()

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
    result_stress_marks: List[str] = []

    for language_code, segment in grouped_segments:
        if language_code is None:
            result.append(segment)
            continue

        if language_code.startswith("en"):
            ipa = english_to_ipa(_normalize_english_segment(segment), keep_punct=False)
        elif language_code == "cmn":
            ipa = hanzi_to_ipa(segment, delimiter='')
        else:
            raise ValueError(f"Unsupported language code: {language_code}")
        ipa = re.sub(r"\s+", "", ipa)
        ipa = re.sub(r'\\p{P}+', '', ipa, flags=re.UNICODE)
        tone_marks = _NON_IPA_TONES_RE.sub(" ", ipa)
        stress_marks = _NON_STRESS_RE.sub(" ", ipa)
        phones = _IPA_TONES_STRESS_RE.sub("", ipa)
        # if language_code == "cmn" and remove_chinese_tone_marks:
        #     ipa = re.sub(r"\d", "", ipa)
        # if language_code.startswith("en") and remove_english_spaces:
        #     ipa = re.sub(r"\s+", "", ipa)
        result.append(phones)
        result_tone_marks.append(tone_marks)
        result_stress_marks.append(stress_marks)
    print(result)
    results = ["".join(result)]
    if get_tone_marks:
        results.append("".join(result_tone_marks))
    if get_stress_marks:
        results.append("".join(result_stress_marks))

    return results[0] if len(results) == 1 else tuple(results)


__all__ = ["text_to_ipa", "LanguageSegmenter"]
