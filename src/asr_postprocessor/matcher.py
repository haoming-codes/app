"""Matching utilities for aligning ASR outputs with canonical terms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .distance import CombinedDistanceConfig, PhoneticDistanceCalculator
from .transcription import PinyinConverter, Syllable, syllables_for_terms


@dataclass
class CandidateTerm:
    """Represents an entity or jargon item that should be protected."""

    surface: str


@dataclass
class MatchResult:
    """Represents a candidate match within the ASR output."""

    candidate: CandidateTerm
    start: int
    end: int
    distance: float


@dataclass
class MatcherConfig:
    """Configuration for candidate matching."""

    max_window_expansion: int = 1
    distance: CombinedDistanceConfig = field(default_factory=CombinedDistanceConfig)
    threshold: float = 0.55


class CandidateMatcher:
    """Find the closest substring match for each candidate term."""

    def __init__(self, candidates: Sequence[CandidateTerm], config: MatcherConfig | None = None) -> None:
        self.candidates = list(candidates)
        self.config = config or MatcherConfig()
        self.converter = PinyinConverter()
        self._candidate_syllables = syllables_for_terms(
            self.converter, (candidate.surface for candidate in self.candidates)
        )
        self._distance = PhoneticDistanceCalculator(self.config.distance)

    def find_matches(self, text: str) -> List[MatchResult]:
        syllables = self.converter.text_to_syllables(text)
        matches: List[MatchResult] = []
        for candidate, candidate_syllables in zip(self.candidates, self._candidate_syllables):
            match = self._best_match_for_candidate(candidate, candidate_syllables, syllables)
            if match is not None:
                matches.append(match)
        return matches

    def _best_match_for_candidate(
        self,
        candidate: CandidateTerm,
        candidate_syllables: Sequence[Syllable],
        syllables: Sequence[Syllable],
    ) -> Optional[MatchResult]:
        target_len = len(candidate_syllables)
        if target_len == 0:
            return None

        best_distance: Optional[float] = None
        best_match: Optional[MatchResult] = None

        min_window = max(1, target_len - self.config.max_window_expansion)
        max_window = target_len + self.config.max_window_expansion
        for window in range(min_window, max_window + 1):
            for start in range(0, len(syllables) - window + 1):
                window_syllables = syllables[start : start + window]
                if not all(s.valid for s in window_syllables):
                    continue
                distance = self._distance.distance(candidate_syllables, window_syllables)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = MatchResult(candidate=candidate, start=start, end=start + window, distance=distance)

        if best_match is None:
            return None

        if best_match.distance > self.config.threshold:
            return None
        return best_match


def apply_matches(text: str, matches: Iterable[MatchResult]) -> str:
    """Apply non-overlapping matches to the text, replacing spans with the candidate surface."""

    sorted_matches = sorted(matches, key=lambda m: (m.distance, m.end - m.start, m.start))
    non_overlapping: List[MatchResult] = []
    occupied: List[tuple[int, int]] = []
    for match in sorted_matches:
        span = (match.start, match.end)
        if any(_overlap(span, existing) for existing in occupied):
            continue
        non_overlapping.append(match)
        occupied.append(span)

    if not non_overlapping:
        return text

    non_overlapping.sort(key=lambda m: m.start)
    result_parts: List[str] = []
    cursor = 0
    for match in non_overlapping:
        result_parts.append(text[cursor:match.start])
        result_parts.append(match.candidate.surface)
        cursor = match.end
    result_parts.append(text[cursor:])
    return "".join(result_parts)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])
