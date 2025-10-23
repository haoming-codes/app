"""Utilities for converting bilingual Chinese/English text to IPA."""
from __future__ import annotations

import re
from dataclasses import dataclass
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
_IPA_VOWEL_RE = re.compile(r"[aeiouɑæɐɜɞəɘɵøœoɔɒʌʊuɯɨʉiɪeɛyʏɜːɝɚ]+")


@dataclass(frozen=True)
class IPAConversionResult:
    phones: List[str]
    tone_marks: List[str]
    stress_marks: List[str]
    syllable_counts: List[int]


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
    return _ALL_CAPS_WORD_RE.sub(lambda match: " ".join(match.group(0)), segment)


def text_to_ipa(
    text: str,
    *,
    segmenter: LanguageSegmenter | None = None,
) -> IPAConversionResult:
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

    phones: List[str] = []
    tone_marks: List[str] = []
    stress_marks: List[str] = []
    syllable_counts: List[int] = []

    for language_code, segment in grouped_segments:
        if language_code is None:
            continue

        if language_code.startswith("en"):
            normalized_segment = _normalize_english_segment(segment)
            cleaned_segment = re.sub(r"\W+", " ", normalized_segment)
            words = [word for word in cleaned_segment.split() if word]
            for word in words:
                ipa = english_to_ipa(word, keep_punct=False)
                ipa = re.sub(r"\s+", "", ipa)
                phones.append(_IPA_TONES_STRESS_RE.sub("", ipa))
                tone_marks.append(_NON_IPA_TONES_RE.sub("", ipa))
                stress_marks.append(_NON_STRESS_RE.sub("", ipa))
                syllable_counts.append(_count_syllables(phones[-1]))
        elif language_code == "cmn":
            for character in segment:
                if _CHINESE_RE.match(character):
                    ipa = hanzi_to_ipa(character, delimiter="")
                    ipa = re.sub(r"\s+", "", ipa)
                    phones.append(_IPA_TONES_STRESS_RE.sub("", ipa))
                    tone_marks.append(_NON_IPA_TONES_RE.sub("", ipa))
                    stress_marks.append(_NON_STRESS_RE.sub("", ipa))
                    syllable_counts.append(1)
                else:
                    continue
        else:
            raise ValueError(f"Unsupported language code: {language_code}")

    return IPAConversionResult(
        phones=phones,
        tone_marks=tone_marks,
        stress_marks=stress_marks,
        syllable_counts=syllable_counts,
    )


def _count_syllables(phones: str) -> int:
    vowel_matches = _IPA_VOWEL_RE.findall(phones)
    count = len(vowel_matches)
    return count or 1


__all__ = ["text_to_ipa", "LanguageSegmenter", "IPAConversionResult"]
