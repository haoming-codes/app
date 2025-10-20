"""Matching logic for phonetic correction of ASR output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .conversion import PhoneticConverter, PhoneticRepresentation
from .distances import DistanceCalculator


@dataclass(frozen=True)
class LexiconEntry:
    """Represents a canonical term expected in the transcription."""

    term: str
    threshold: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    """Stores the outcome of a fuzzy phonetic match."""

    lexicon_entry: LexiconEntry
    start: int
    end: int
    original: str
    distance: float

    @property
    def replacement(self) -> str:
        return self.lexicon_entry.term


class PhoneticMatcher:
    """Matches substrings against a lexicon using phonetic distance."""

    def __init__(
        self,
        converter: Optional[PhoneticConverter] = None,
        distance_calculator: Optional[DistanceCalculator] = None,
        default_threshold: float = 0.6,
    ) -> None:
        self.converter = converter or PhoneticConverter()
        if distance_calculator is None:
            raise ValueError("A DistanceCalculator must be provided.")
        self.distance_calculator = distance_calculator
        self.default_threshold = default_threshold
        self._cache: Dict[str, PhoneticRepresentation] = {}

    def match(self, text: str, lexicon: Iterable[LexiconEntry]) -> List[MatchResult]:
        matches: List[MatchResult] = []
        for entry in lexicon:
            entry_repr = self._representation(entry.term)
            window_length = len(entry.term)
            if window_length == 0:
                continue
            for start in range(0, len(text) - window_length + 1):
                end = start + window_length
                candidate = text[start:end]
                candidate_repr = self._representation(candidate)
                if not candidate_repr.ipa or len(candidate_repr.ipa) != len(entry_repr.ipa):
                    continue
                distance = self.distance_calculator.distance(
                    candidate_repr.toneless_ipa(),
                    entry_repr.toneless_ipa(),
                    candidate_repr.tones,
                    entry_repr.tones,
                )
                threshold = entry.threshold if entry.threshold is not None else self.default_threshold
                if distance <= threshold:
                    matches.append(
                        MatchResult(
                            lexicon_entry=entry,
                            start=start,
                            end=end,
                            original=candidate,
                            distance=distance,
                        )
                    )
        matches.sort(key=lambda result: result.distance)
        return matches

    def correct_text(self, text: str, lexicon: Iterable[LexiconEntry]) -> tuple[str, List[MatchResult]]:
        matches = self.match(text, lexicon)
        if not matches:
            return text, []
        occupied = [False] * len(text)
        accepted: List[MatchResult] = []
        for match in matches:
            if any(occupied[i] for i in range(match.start, match.end)):
                continue
            for i in range(match.start, match.end):
                occupied[i] = True
            accepted.append(match)
        accepted.sort(key=lambda result: result.start)
        corrected_segments: List[str] = []
        cursor = 0
        for match in accepted:
            corrected_segments.append(text[cursor:match.start])
            corrected_segments.append(match.replacement)
            cursor = match.end
        corrected_segments.append(text[cursor:])
        corrected_text = "".join(corrected_segments)
        return corrected_text, accepted

    def _representation(self, text: str) -> PhoneticRepresentation:
        cached = self._cache.get(text)
        if cached is None:
            cached = self.converter.convert(text)
            self._cache[text] = cached
        return cached
