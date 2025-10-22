"""Conversion utilities for Chinese/English bilingual text."""
from __future__ import annotations

import re
from typing import List, Tuple

from phonemizer import phonemize

_CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF]")
_ENGLISH_CHAR_RE = re.compile(r"[A-Za-z]")


def _character_category(character: str) -> str:
    """Return the language category for *character*.

    The function distinguishes between Chinese characters, English letters, and
    other characters such as whitespace or punctuation.
    """

    if _CHINESE_CHAR_RE.match(character):
        return "chinese"
    if _ENGLISH_CHAR_RE.match(character):
        return "english"
    return "other"


def _split_text(text: str) -> List[Tuple[str, str]]:
    """Split *text* into contiguous language-specific segments.

    Returns a list of ``(category, segment)`` tuples where *category* is one of
    ``"chinese"``, ``"english"``, or ``"other"``.
    """

    segments: List[Tuple[str, str]] = []
    current_category: str | None = None
    current_segment: List[str] = []

    for character in text:
        category = _character_category(character)

        if category == "other":
            if current_segment:
                segments.append((current_category or "other", "".join(current_segment)))
                current_segment = []
                current_category = None
            segments.append(("other", character))
            continue

        if category != current_category and current_segment:
            segments.append((current_category or category, "".join(current_segment)))
            current_segment = []

        current_category = category
        current_segment.append(character)

    if current_segment:
        segments.append((current_category or "other", "".join(current_segment)))

    return segments


def text_to_ipa(text: str, **phonemize_kwargs: str) -> str:
    """Convert bilingual Chinese/English *text* to an IPA transcription.

    The function splits the text into Chinese and English segments and
    phonemizes each portion independently with :mod:`phonemizer` using the
    ``espeak-ng`` backend. Keyword arguments are forwarded to
    :func:`phonemizer.phonemize` to allow callers to customise its behaviour.

    Characters that do not belong to either language (for example punctuation)
    are preserved without modification.
    """

    if not text:
        return ""

    backend = phonemize_kwargs.setdefault("backend", "espeak-ng")
    _ = backend  # silence linters about unused variable when inspected

    segments = _split_text(text)
    converted: List[str] = []

    for category, segment in segments:
        if category == "chinese":
            converted.append(phonemize(segment, language="cmn", **phonemize_kwargs))
        elif category == "english":
            converted.append(phonemize(segment, language="en", **phonemize_kwargs))
        else:
            converted.append(segment)

    return "".join(converted)


__all__ = ["text_to_ipa"]
