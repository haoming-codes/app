"""Utilities for converting bilingual Chinese/English text to IPA."""
from __future__ import annotations

import re
from typing import Dict, List, TYPE_CHECKING

try:  # pragma: no cover - exercised indirectly via public API
    import epitran as _epitran_module
except ModuleNotFoundError:  # pragma: no cover - handled in _get_transliterator
    _epitran_module = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
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
            return "cmn"
        if _ENGLISH_RE.match(segment):
            return "en-us"
        return None
_EPITRAN_CONFIG: Dict[str, dict[str, object]] = {
    "cmn": {"code": "cmn-Hans", "kwargs": {"cedict_file": "cedict_ts.u8", "tones": True}},
    "en-us": {"code": "eng-Latn", "kwargs": {}},
}

_EPITRAN_CACHE: Dict[str, "epitran.Epitran"] = {}


def _get_transliterator(language_code: str) -> "epitran.Epitran":
    try:
        config = _EPITRAN_CONFIG[language_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported language code: {language_code}") from exc

    if _epitran_module is None:
        raise ModuleNotFoundError("epitran is required to transliterate text")

    if language_code not in _EPITRAN_CACHE:
        _EPITRAN_CACHE[language_code] = _epitran_module.Epitran(config["code"], **config["kwargs"])

    return _EPITRAN_CACHE[language_code]


def _validate_kwargs(kwargs: dict[str, object]) -> None:
    if "language" in kwargs:
        raise ValueError("The 'language' argument is determined automatically and must not be provided.")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"Unexpected keyword arguments for transliteration: {unexpected}")


def text_to_ipa(
    text: str,
    *,
    segmenter: LanguageSegmenter | None = None,
    get_tone_marks: bool = False,
    **transliterate_kwargs,
) -> str:
    """Convert Chinese/English text into IPA using Epitran."""
    if segmenter is None:
        segmenter = LanguageSegmenter()

    _validate_kwargs(transliterate_kwargs)

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

        transliterator = _get_transliterator(language_code)
        ipa = transliterator.transliterate(segment)
        tone_marks = re.sub(r"\D", " ", ipa)
        phones = re.sub(r"\d", "", ipa)
        result.append(phones)
        result_tone_marks.append(tone_marks)

    results = ["".join(result)]
    if get_tone_marks:
        results.append("".join(result_tone_marks))

    return results[0] if len(results) == 1 else tuple(results)


__all__ = ["text_to_ipa", "LanguageSegmenter"]
