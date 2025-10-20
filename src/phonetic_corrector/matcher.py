"""Phonetic fuzzy matching of named entities within ASR output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .distance import PhoneticDistance
from .encoder import PhoneticEncoder, PhoneticEncoding


@dataclass(frozen=True)
class MatchCandidate:
    """Represents a candidate correction for a span of the ASR output."""

    entity: str
    start: int
    end: int
    score: float
    original: str


@dataclass(frozen=True)
class _EntityEncoding:
    entity: str
    segments: Sequence[str]
    length: int


class PhoneticEntityMatcher:
    """Match named entities in ASR output using phonetic similarity."""

    def __init__(
        self,
        entities: Iterable[str],
        *,
        threshold: float = 0.6,
    ) -> None:
        if threshold <= 0 or threshold > 1:  # pragma: no cover - defensive
            raise ValueError("threshold must be within (0, 1]")

        self._encoder = PhoneticEncoder()
        self._distance = PhoneticDistance()
        self._threshold = threshold
        self._entities: List[_EntityEncoding] = []
        for entity in entities:
            if not entity:
                continue
            encoding = self._encoder.encode(entity)
            self._entities.append(
                _EntityEncoding(entity=entity, segments=encoding.segments, length=len(entity))
            )

    def find_matches(self, transcription: str) -> List[MatchCandidate]:
        """Return phonetic match candidates ordered by their span."""

        encoded = self._encoder.encode(transcription)
        candidates = self._collect_candidates(encoded)
        resolved = self._resolve_conflicts(candidates)
        return sorted(resolved, key=lambda candidate: candidate.start)

    def _collect_candidates(self, encoded: PhoneticEncoding) -> List[MatchCandidate]:
        candidates: List[MatchCandidate] = []
        for entity in self._entities:
            for start in range(0, len(encoded.text) - entity.length + 1):
                end = start + entity.length
                window = encoded.window(start, end)
                score = self._distance.similarity(entity.segments, window)
                if score < self._threshold:
                    continue
                original = encoded.text[start:end]
                candidates.append(
                    MatchCandidate(
                        entity=entity.entity,
                        start=start,
                        end=end,
                        score=score,
                        original=original,
                    )
                )
        return candidates

    @staticmethod
    def _resolve_conflicts(candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        occupied = [False] * max(candidate.end for candidate in candidates)
        selected: List[MatchCandidate] = []
        for candidate in candidates:
            if any(occupied[candidate.start : candidate.end]):
                continue
            for index in range(candidate.start, candidate.end):
                occupied[index] = True
            selected.append(candidate)
        return selected


__all__ = ["PhoneticEntityMatcher", "MatchCandidate"]
