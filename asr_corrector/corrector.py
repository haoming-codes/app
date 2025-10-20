"""Phonetic correction utilities for Chinese ASR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from panphon.distance import Distance

from .entities import NameEntity
from .phonetics import hanzi_to_ipa, syllables


@dataclass(frozen=True)
class CorrectionMatch:
    """Represents a single phonetic match in the transcript."""

    start: int
    end: int
    score: float
    original: str
    entity: NameEntity


@dataclass(frozen=True)
class CorrectionResult:
    """The outcome of a correction run."""

    corrected: str
    matches: Tuple[CorrectionMatch, ...]


@dataclass(frozen=True)
class _PreparedEntity:
    entity: NameEntity
    ipa: str
    ipa_compact: str
    syllable_count: int


class PhoneticCorrector:
    """Correct ASR outputs by comparing IPA representations of substrings."""

    def __init__(
        self,
        entities: Iterable[NameEntity],
        *,
        threshold: float = 0.35,
        length_tolerance: int = 1,
    ) -> None:
        """Create a corrector.

        Parameters
        ----------
        entities
            Canonical name entities against which the transcript will be matched.
        threshold
            Maximum normalized phonetic distance to accept a correction.
        length_tolerance
            How many additional characters above or below the entity syllable
            count should be considered when scanning for candidates. Use this to
            compensate for insertion/deletion errors from the ASR model.
        """

        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if length_tolerance < 0:
            raise ValueError("length_tolerance must be non-negative")

        self._distance = Distance()
        self._threshold = threshold
        self._length_tolerance = length_tolerance
        self._prepared_entities = tuple(self._prepare_entity(entity) for entity in entities)

    def _prepare_entity(self, entity: NameEntity) -> _PreparedEntity:
        ipa = entity.ipa or hanzi_to_ipa(entity.canonical)
        prepared = _PreparedEntity(
            entity=entity.with_ipa(ipa),
            ipa=ipa,
            ipa_compact=ipa.replace(" ", ""),
            syllable_count=max(len(syllables(entity.canonical)), 1),
        )
        return prepared

    def _normalized_distance(self, ipa_a: str, ipa_b: str) -> float:
        ipa_a_compact = ipa_a.replace(" ", "")
        ipa_b_compact = ipa_b.replace(" ", "")
        max_len = max(len(ipa_a_compact), len(ipa_b_compact), 1)
        distance = self._distance.weighted_feature_edit_distance(ipa_a_compact, ipa_b_compact)
        return distance / max_len

    def find_matches(self, transcript: str) -> Tuple[CorrectionMatch, ...]:
        """Return all non-overlapping matches in the transcript."""

        if not transcript:
            return tuple()

        matches: List[CorrectionMatch] = []

        for prepared in self._prepared_entities:
            lengths = range(
                max(1, prepared.syllable_count - self._length_tolerance),
                prepared.syllable_count + self._length_tolerance + 1,
            )
            for length in lengths:
                for start in range(0, len(transcript) - length + 1):
                    end = start + length
                    candidate = transcript[start:end]
                    ipa = hanzi_to_ipa(candidate)
                    score = self._normalized_distance(ipa, prepared.ipa)
                    if score <= self._threshold:
                        matches.append(
                            CorrectionMatch(
                                start=start,
                                end=end,
                                score=score,
                                original=candidate,
                                entity=prepared.entity,
                            )
                        )

        # Select best non-overlapping matches greedily by score (lower is better)
        matches.sort(key=lambda match: match.score)
        occupied = [False] * len(transcript)
        chosen: List[CorrectionMatch] = []
        for match in matches:
            if any(occupied[i] for i in range(match.start, match.end)):
                continue
            for i in range(match.start, match.end):
                occupied[i] = True
            chosen.append(match)

        return tuple(sorted(chosen, key=lambda m: m.start))

    def correct(self, transcript: str) -> CorrectionResult:
        """Return the corrected transcript and the matches that triggered it."""

        if not transcript:
            return CorrectionResult(corrected=transcript, matches=tuple())

        matches = self.find_matches(transcript)
        if not matches:
            return CorrectionResult(corrected=transcript, matches=matches)

        pieces: List[str] = []
        cursor = 0
        for match in matches:
            pieces.append(transcript[cursor:match.start])
            pieces.append(match.entity.canonical)
            cursor = match.end
        pieces.append(transcript[cursor:])
        corrected = "".join(pieces)
        return CorrectionResult(corrected=corrected, matches=matches)
