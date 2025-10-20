"""Utilities for matching substrings against a pronunciation dictionary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import DEFAULT_DISTANCE_CONFIG, DistanceConfig
from .distance import PhoneticDistanceCalculator
from .transcription import _contains_chinese


@dataclass(frozen=True)
class MatchResult:
    """Record describing a phonetic match candidate."""

    substring: str
    candidate: str
    distance: float
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _split_units(text: str) -> List[str]:
    if _contains_chinese(text):
        return list(text)
    return text.split()


def _join_units(units: Iterable[str], chinese: bool) -> str:
    if chinese:
        return "".join(units)
    return " ".join(units)


class PhoneticMatcher:
    """Perform phonetic substring matching with configurable distance metrics."""

    def __init__(self, dictionary: Sequence[str], config: DistanceConfig | None = None) -> None:
        self.dictionary = list(dictionary)
        self.config = deepcopy(config) if config is not None else deepcopy(DEFAULT_DISTANCE_CONFIG)
        self.calculator = PhoneticDistanceCalculator(self.config)

    def _threshold(self, threshold: float | None) -> float:
        return threshold if threshold is not None else self.config.threshold

    def match(self, text: str, threshold: float | None = None) -> List[MatchResult]:
        chinese = _contains_chinese(text)
        units = _split_units(text)
        matches: List[MatchResult] = []
        if not units:
            return matches
        threshold_value = self._threshold(threshold)
        for entry in self.dictionary:
            entry_units = _split_units(entry)
            target_len = max(len(entry_units), 1)
            for start in range(0, len(units) - target_len + 1):
                end = start + target_len
                candidate_units = units[start:end]
                candidate_text = _join_units(candidate_units, chinese)
                distance = self.calculator.distance(candidate_text, entry)
                if distance <= threshold_value:
                    matches.append(
                        MatchResult(
                            substring=candidate_text,
                            candidate=entry,
                            distance=distance,
                            start=start,
                            end=end,
                        )
                    )
        matches.sort(key=lambda result: result.distance)
        return matches

    def distance(self, a: str, b: str) -> float:
        return self.calculator.distance(a, b)
