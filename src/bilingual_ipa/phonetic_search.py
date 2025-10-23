"""Sliding-window phonetic search utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .converter import IPAConversionResult, text_to_ipa
from .phone_distance import AggregateStrategy, phone_distance


@dataclass(frozen=True)
class WindowDistance:
    """Represents the phonetic distance for a window within a sentence."""

    start_index: int
    """The inclusive index of the first phone in the window."""

    end_index: int
    """The exclusive index of the last phone in the window."""

    phones: str
    """Concatenated IPA phones for the window."""

    syllable_count: int
    """Number of syllables covered by the window."""

    distance: float
    """Aggregated phonetic distance to the query phrase."""


def _ensure_non_empty(result: IPAConversionResult, *, label: str) -> None:
    if not result.phones:
        raise ValueError(f"{label} produced no convertible IPA phones.")


def _phrase_ipa(phrase: str) -> tuple[str, int]:
    phrase_result = text_to_ipa(phrase)
    _ensure_non_empty(phrase_result, label="Phrase")
    combined = "".join(phrase_result.phones)
    syllable_count = sum(phrase_result.syllable_counts)
    if syllable_count <= 0:
        raise ValueError("Phrase must contain at least one syllable.")
    return combined, syllable_count


def _iter_windows(
    syllable_counts: Sequence[int],
    target_syllables: int,
    tolerance: int,
) -> Iterable[tuple[int, int, int]]:
    """Yield window boundaries whose syllable counts are near the target."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    n = len(syllable_counts)
    for start in range(n):
        cumulative = 0
        for end in range(start, n):
            cumulative += syllable_counts[end]
            if cumulative > target_syllables + tolerance:
                break
            if abs(cumulative - target_syllables) <= tolerance:
                yield start, end + 1, cumulative


def window_phonetic_distances(
    sentence: str,
    phrase: str,
    *,
    metrics: Iterable[str] | str | None = None,
    weights: Mapping[str, float] | None = None,
    aggregate: AggregateStrategy = "mean",
    syllable_tolerance: int = 1,
) -> list[WindowDistance]:
    """Compute phonetic distances between a query phrase and windows in a sentence.

    The sentence is converted to IPA using :func:`text_to_ipa`. Sliding windows of
    phones are evaluated where the total number of syllables is close to the
    number of syllables of ``phrase`` (within ``syllable_tolerance``). Each
    window's phones are joined before calculating the phonetic distance to the
    joined phones of the query phrase using :func:`phone_distance`.

    Args:
        sentence: The sentence containing potential matches.
        phrase: The query phrase to compare against the sentence windows.
        metrics: Optional set of metric names to pass to :func:`phone_distance`.
        weights: Optional metric weights supplied to :func:`phone_distance`.
        aggregate: Aggregation strategy passed to :func:`phone_distance`.
        syllable_tolerance: Allowed difference between the number of syllables in
            a window and the query phrase.

    Returns:
        A list of :class:`WindowDistance` entries ordered by their appearance in
        the sentence. If the sentence contains no convertible phones, the list is
        empty.

    Raises:
        ValueError: If the phrase cannot be converted to IPA or contains no
            syllables, or if ``syllable_tolerance`` is negative.
    """

    phrase_phone, phrase_syllables = _phrase_ipa(phrase)

    sentence_result = text_to_ipa(sentence)
    if not sentence_result.phones:
        return []

    windows: list[WindowDistance] = []
    for start, end, syllables in _iter_windows(
        sentence_result.syllable_counts, phrase_syllables, syllable_tolerance
    ):
        window_phones = "".join(sentence_result.phones[start:end])
        distance = phone_distance(
            window_phones,
            phrase_phone,
            metrics,
            weights=weights,
            aggregate=aggregate,
        )
        windows.append(
            WindowDistance(
                start_index=start,
                end_index=end,
                phones=window_phones,
                syllable_count=syllables,
                distance=distance,
            )
        )

    return windows


__all__ = ["WindowDistance", "window_phonetic_distances"]

