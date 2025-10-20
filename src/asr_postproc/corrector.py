from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import panphon.distance

from .entities import EntityMatch, EntitySpec, build_entity_specs, text_to_ipa_sequence


@dataclass
class CorrectionResult:
    """Result of running entity correction over an ASR transcript."""

    original_text: str
    corrected_text: str
    matches: List[EntityMatch]


class PhoneticEntityCorrector:
    """Correct named entities in Chinese ASR transcripts.

    The corrector uses IPA sequences derived from :mod:`pypinyin` and computes
    phonetic similarity scores via :mod:`panphon`.  It scans the transcript for
    substrings whose phonetic representation is sufficiently similar to the
    canonical surface form of the entities provided at initialization time.
    """

    def __init__(
        self,
        entities: Iterable[str | EntitySpec],
        *,
        similarity_threshold: float = 0.65,
        max_length_delta: int = 1,
    ) -> None:
        """Create a new corrector.

        Parameters
        ----------
        entities
            Iterable of canonical entity surface forms or :class:`EntitySpec`
            objects.
        similarity_threshold
            Minimum similarity required to accept a candidate match.  The score
            ranges from ``0`` (no similarity) to ``1`` (identical IPA features).
        max_length_delta
            When scanning for matches, substrings whose length differs from the
            entity length by at most this value are considered.  This helps
            handle common insertion or deletion errors in ASR output.
        """

        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in the range [0, 1]")
        if max_length_delta < 0:
            raise ValueError("max_length_delta must be non-negative")

        self._entities: List[EntitySpec] = build_entity_specs(entities)
        self._similarity_threshold = similarity_threshold
        self._max_length_delta = max_length_delta
        self._distance = panphon.distance.Distance()

    def _similarity(self, ipa_a: Sequence[str], ipa_b: Sequence[str]) -> float:
        if not ipa_a and not ipa_b:
            return 1.0
        ipa_string_a = " ".join(ipa_a)
        ipa_string_b = " ".join(ipa_b)
        dist = self._distance.feature_edit_distance(ipa_string_a, ipa_string_b)
        norm = max(len(ipa_a), len(ipa_b), 1)
        similarity = 1.0 - (dist / norm)
        if similarity < 0.0:
            similarity = 0.0
        return similarity

    def _best_match_for_entity(
        self, text: str, ipa_cache: List[List[str]], entity: EntitySpec
    ) -> Tuple[float, Tuple[int, int]]:
        best_score = -1.0
        best_span = (-1, -1)
        entity_length = entity.length
        for delta in range(-self._max_length_delta, self._max_length_delta + 1):
            candidate_length = entity_length + delta
            if candidate_length <= 0:
                continue
            for start in range(0, len(text) - candidate_length + 1):
                end = start + candidate_length
                ipa_tokens = ipa_cache[start:end]
                flattened: List[str] = []
                for tokens in ipa_tokens:
                    flattened.extend(tokens)
                score = self._similarity(flattened, entity.canonical_ipa)
                if score > best_score:
                    best_score = score
                    best_span = (start, end)
        return best_score, best_span

    def _build_ipa_cache(self, text: str) -> List[List[str]]:
        cache: List[List[str]] = []
        for char in text:
            cache.append(text_to_ipa_sequence(char))
        return cache

    def correct(self, text: str) -> CorrectionResult:
        """Apply entity corrections to the given ASR transcript."""

        if not text:
            return CorrectionResult(text, text, [])

        ipa_cache = self._build_ipa_cache(text)
        matches: List[EntityMatch] = []
        for entity in self._entities:
            score, span = self._best_match_for_entity(text, ipa_cache, entity)
            if score < self._similarity_threshold:
                continue
            start, end = span
            observed = text[start:end]
            matches.append(
                EntityMatch(
                    entity=entity,
                    start=start,
                    end=end,
                    similarity=score,
                    observed=observed,
                )
            )

        resolved_matches = self._resolve_conflicts(matches)
        corrected_text = self._apply_matches(text, resolved_matches)
        return CorrectionResult(text, corrected_text, resolved_matches)

    def _resolve_conflicts(self, matches: Sequence[EntityMatch]) -> List[EntityMatch]:
        ordered = sorted(matches, key=lambda m: m.similarity, reverse=True)
        accepted: List[EntityMatch] = []
        for match in ordered:
            if any(self._overlaps(match, other) for other in accepted):
                continue
            accepted.append(match)
        accepted.sort(key=lambda m: m.start)
        return accepted

    @staticmethod
    def _overlaps(match_a: EntityMatch, match_b: EntityMatch) -> bool:
        return not (match_a.end <= match_b.start or match_b.end <= match_a.start)

    def _apply_matches(self, text: str, matches: Sequence[EntityMatch]) -> str:
        if not matches:
            return text
        result = []
        cursor = 0
        for match in matches:
            if cursor < match.start:
                result.append(text[cursor:match.start])
            result.append(match.replacement())
            cursor = match.end
        if cursor < len(text):
            result.append(text[cursor:])
        return "".join(result)
