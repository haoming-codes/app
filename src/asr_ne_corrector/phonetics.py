"""Utilities for converting Chinese text to IPA and computing similarity."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

try:
    from pypinyin import Style, pinyin
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise RuntimeError(
        "pypinyin is required to use the phonetic conversion utilities."
    ) from exc

try:
    from panphon.distance import Distance
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise RuntimeError(
        "panphon is required to compute IPA-based similarity scores."
    ) from exc


_DISTANCE = Distance()


def text_to_ipa(text: str) -> str:
    """Convert Chinese text to a whitespace-delimited IPA transcription."""

    if not text:
        return ""

    segments: Sequence[Sequence[str]] = pinyin(
        text,
        style=Style.IPA,
        heteronym=False,
        errors="ignore",
        strict=False,
    )
    flattened: Iterable[str] = (segment for group in segments for segment in group if segment)
    ipa = " ".join(flattened)
    return ipa


@lru_cache(maxsize=1024)
def ipa_similarity(ipa_a: str, ipa_b: str) -> float:
    """Compute a similarity score between two IPA strings.

    The score is :math:`1 - d`, where ``d`` is the normalized feature edit distance
    between the two strings. A return value of ``1.0`` indicates an exact match
    and ``0.0`` indicates the maximum possible distance.
    """

    if not ipa_a or not ipa_b:
        return 0.0

    distance = _DISTANCE.normalized_distance(ipa_a, ipa_b)
    return max(0.0, 1.0 - distance)


def text_similarity(text_a: str, text_b: str) -> float:
    """Convenience wrapper that converts inputs to IPA before scoring."""

    return ipa_similarity(text_to_ipa(text_a), text_to_ipa(text_b))
