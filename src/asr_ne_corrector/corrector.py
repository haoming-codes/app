"""Named entity correction driven by phonetic similarity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .entities import EntityMatch, NameEntity
from .phonetics import ipa_similarity, text_to_ipa


@dataclass
class _CandidateSubstring:
    start: int
    end: int
    text: str
    ipa: str


class NameEntityCorrector:
    """Suggest replacements for named entities in noisy ASR transcripts."""

    def __init__(
        self,
        entities: Iterable[str],
        *,
        similarity_threshold: float = 0.6,
        length_tolerance: int = 1,
    ) -> None:
        """Create a corrector for the provided entities.

        Parameters
        ----------
        entities:
            Iterable of canonical entity names written in Chinese.
        similarity_threshold:
            Minimum IPA similarity required to propose a replacement. Must be in
            the interval ``[0.0, 1.0]``.
        length_tolerance:
            Maximum absolute difference (in characters) between the candidate
            substring and the canonical entity. Increasing this value leads to a
            larger search space and slower matching, but allows insertions or
            deletions in the ASR output to be recovered.
        """

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if length_tolerance < 0:
            raise ValueError("length_tolerance must be non-negative")

        self.similarity_threshold = similarity_threshold
        self.length_tolerance = length_tolerance

        self.entities: List[NameEntity] = [
            NameEntity(text=entity.strip(), ipa=text_to_ipa(entity.strip()))
            for entity in entities
            if entity and entity.strip()
        ]

        self._entities_by_length: Dict[int, List[NameEntity]] = defaultdict(list)
        for entity in self.entities:
            if not entity.ipa:
                continue
            self._entities_by_length[entity.length].append(entity)

        self._entity_lengths: Sequence[int] = sorted(self._entities_by_length)

    # ------------------------------------------------------------------
    def suggest(self, text: str, *, top_k: int = 5) -> List[EntityMatch]:
        """Return the best matching entity replacements for ``text``."""

        matches = self._find_matches(text)
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------
    def correct(self, text: str) -> Tuple[str, List[EntityMatch]]:
        """Return ``text`` with high-confidence matches replaced."""

        matches = self._find_matches(text)
        matches.sort(key=lambda match: match.score, reverse=True)

        chosen: List[EntityMatch] = []
        occupied: List[Tuple[int, int]] = []

        for match in matches:
            if any(not (match.end <= start or match.start >= end) for start, end in occupied):
                continue
            chosen.append(match)
            occupied.append((match.start, match.end))

        if not chosen:
            return text, []

        chosen.sort(key=lambda match: match.start)
        corrected: List[str] = []
        cursor = 0
        for match in chosen:
            corrected.append(text[cursor : match.start])
            corrected.append(match.replacement())
            cursor = match.end
        corrected.append(text[cursor:])

        corrected_text = "".join(corrected)
        return corrected_text, chosen

    # ------------------------------------------------------------------
    def _candidate_lengths(self, length: int) -> Sequence[int]:
        return [
            entity_length
            for entity_length in self._entity_lengths
            if abs(entity_length - length) <= self.length_tolerance
        ]

    def _generate_substrings(self, text: str) -> Iterable[_CandidateSubstring]:
        n = len(text)
        if n == 0:
            return []

        min_len = max(1, min(self._entity_lengths, default=1) - self.length_tolerance)
        max_len = max(self._entity_lengths, default=1) + self.length_tolerance

        for start in range(n):
            for length in range(min_len, max_len + 1):
                end = start + length
                if end > n:
                    break
                substring = text[start:end]
                ipa = text_to_ipa(substring)
                if not ipa:
                    continue
                yield _CandidateSubstring(start=start, end=end, text=substring, ipa=ipa)

    def _find_matches(self, text: str) -> List[EntityMatch]:
        matches: List[EntityMatch] = []

        if not self.entities or not text:
            return matches

        for candidate in self._generate_substrings(text):
            candidate_lengths = self._candidate_lengths(candidate.end - candidate.start)
            if not candidate_lengths:
                continue

            for entity_length in candidate_lengths:
                for entity in self._entities_by_length.get(entity_length, []):
                    score = ipa_similarity(candidate.ipa, entity.ipa)
                    if score < self.similarity_threshold:
                        continue
                    matches.append(
                        EntityMatch(
                            start=candidate.start,
                            end=candidate.end,
                            observed=candidate.text,
                            entity=entity,
                            score=score,
                        )
                    )
        return matches
