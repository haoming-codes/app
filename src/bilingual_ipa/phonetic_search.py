"""Sliding-window phonetic search utilities."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence

from .conversion import IPAConversionResult, text_to_ipa
from .distances import (
    CompositeDistanceCalculator,
    DistanceCalculator,
    PhoneDistanceCalculator,
    ToneDistanceCalculator,
)


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

    phrase: str | None = None
    """Phrase compared against the window, when available."""


def _ensure_non_empty(result: IPAConversionResult, *, label: str) -> None:
    if not result.phones:
        raise ValueError(f"{label} produced no convertible IPA phones.")


def _phrase_result(phrase: str) -> tuple[IPAConversionResult, int]:
    phrase_result = text_to_ipa(phrase)
    _ensure_non_empty(phrase_result, label="Phrase")
    syllable_count = sum(phrase_result.syllable_counts)
    if syllable_count <= 0:
        raise ValueError("Phrase must contain at least one syllable.")
    return phrase_result, syllable_count


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
    distance_calculator: DistanceCalculator | None = None,
    syllable_tolerance: int = 1,
) -> list[WindowDistance]:
    """Compute phonetic distances between a query phrase and windows in a sentence.

    The sentence is converted to IPA using :func:`text_to_ipa`. Sliding windows of
    phones are evaluated where the total number of syllables is close to the
    number of syllables of ``phrase`` (within ``syllable_tolerance``). Each
    window's phones are joined before calculating the phonetic distance to the
    query phrase using the provided :class:`~bilingual_ipa.distances.DistanceCalculator`.

    Args:
        sentence: The sentence containing potential matches.
        phrase: The query phrase to compare against the sentence windows.
        distance_calculator: Distance calculator used to compare the phrase and
            each window. If omitted, a composite calculator that combines phone
            and tone distances is used.
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

    phrase_result, phrase_syllables = _phrase_result(phrase)

    sentence_result = text_to_ipa(sentence)
    if not sentence_result.phones:
        return []

    if distance_calculator is None:
        distance_calculator = CompositeDistanceCalculator(
            [PhoneDistanceCalculator(), ToneDistanceCalculator()],
            aggregate="sum",
        )

    windows: list[WindowDistance] = []
    for start, end, syllables in _iter_windows(
        sentence_result.syllable_counts, phrase_syllables, syllable_tolerance
    ):
        window_phones = "".join(sentence_result.phones[start:end])
        window_result = IPAConversionResult(
            phones=list(sentence_result.phones[start:end]),
            tone_marks=list(sentence_result.tone_marks[start:end]),
            stress_marks=list(sentence_result.stress_marks[start:end]),
            syllable_counts=list(sentence_result.syllable_counts[start:end]),
            tokens=list(sentence_result.tokens[start:end]),
        )
        distance = distance_calculator.distance(window_result, phrase_result)
        windows.append(
            WindowDistance(
                start_index=start,
                end_index=end,
                phones=window_phones,
                syllable_count=syllables,
                distance=distance,
                phrase=phrase,
            )
        )

    return windows


class PhoneticWindowRetriever:
    """Retrieve phonetic window matches for a sentence against a vocabulary."""

    def __init__(
        self,
        *,
        distance_calculator: DistanceCalculator | None = None,
        syllable_tolerance: int = 1,
    ) -> None:
        if distance_calculator is None:
            distance_calculator = CompositeDistanceCalculator(
                [PhoneDistanceCalculator(), ToneDistanceCalculator()],
                aggregate="sum",
            )
        self._distance_calculator = distance_calculator
        self._syllable_tolerance = syllable_tolerance
        self._results: list[WindowDistance] = []

    @property
    def results(self) -> Sequence[WindowDistance]:
        """Return the latest computed distances."""

        return self._results

    def compute_all_distances(self, sentence: str, vocabulary: Iterable[str]) -> list[WindowDistance]:
        """Compute distances between ``sentence`` and each phrase in ``vocabulary``."""

        distances: list[WindowDistance] = []
        for phrase in vocabulary:
            for window in window_phonetic_distances(
                sentence,
                phrase,
                distance_calculator=self._distance_calculator,
                syllable_tolerance=self._syllable_tolerance,
            ):
                distances.append(window)

        self._results = sorted(distances, key=lambda wd: wd.distance)
        return self._results

    def top_k(self, k: int) -> list[WindowDistance]:
        """Return the ``k`` closest windows to the vocabulary phrases."""

        if k < 0:
            raise ValueError("k must be non-negative")
        return list(self._results[:k]) if k else []

    def within_threshold(self, threshold: float) -> list[WindowDistance]:
        """Return all windows whose distance is below ``threshold``."""

        return [window for window in self._results if window.distance <= threshold]


class PhoneticWindowRewriter(PhoneticWindowRetriever):
    """Retrieve phonetic matches and rewrite the original transcription."""

    def retrieve_and_rewrite(
        self,
        sentence: str,
        vocabulary: Iterable[str],
        *,
        threshold: float,
    ) -> list[str]:
        """Return the tokenized transcription with the closest match rewritten.

        The sentence is converted to IPA to recover the original tokenization. The
        method computes phonetic distances against each phrase in ``vocabulary``
        and rewrites the sentence tokens using the closest match when its distance
        falls below ``threshold``. If no acceptable match exists, the original
        tokenization is returned unchanged.
        """

        sentence_result = text_to_ipa(sentence)
        if not sentence_result.tokens:
            return list(sentence_result.tokens)

        results = self.compute_all_distances(sentence, vocabulary)
        if not results:
            return list(sentence_result.tokens)

        best = results[0]
        if best.phrase is None:
            return list(sentence_result.tokens)

        if best.distance > threshold:
            return list(sentence_result.tokens)

        if len(results) > 1 and results[1].distance <= best.distance:
            return list(sentence_result.tokens)

        rewritten = list(sentence_result.tokens)
        rewritten[best.start_index : best.end_index] = [best.phrase]
        return rewritten


__all__ = [
    "WindowDistance",
    "window_phonetic_distances",
    "PhoneticWindowRetriever",
    "PhoneticWindowRewriter",
]

