"""Sliding-window matching utilities."""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

from .config import DistanceConfig, KnowledgeBaseEntry, MatcherConfig
from .distance import DistanceBreakdown, PhoneticDistanceCalculator


@dataclass
class CandidateCorrection:
    """Represents a candidate correction for a substring."""

    substring: str
    entry: KnowledgeBaseEntry
    distance: DistanceBreakdown


def tokenize(text: str) -> List[str]:
    """Tokenize a mixed Chinese/English string."""

    tokens: List[str] = []
    buffer: List[str] = []
    current_is_cjk = None

    for char in text:
        is_cjk = "\u4e00" <= char <= "\u9fff"
        if current_is_cjk is None:
            current_is_cjk = is_cjk
        if is_cjk != current_is_cjk or char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            current_is_cjk = is_cjk
        if not char.isspace():
            buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


class WindowMatcher:
    """Perform sliding-window matching against the knowledge base."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config
        self.distance_calculator = PhoneticDistanceCalculator(config.segmental_language_map)

    def generate_windows(self, tokens: Sequence[str], size: int) -> Iterator[Tuple[int, int, str]]:
        for start in range(len(tokens) - size + 1):
            end = start + size
            yield start, end, "".join(tokens[start:end])

    def language_for(self, text: str) -> str:
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return "zh"
        return "en"

    def match(
        self,
        text: str,
        kb_entries: Sequence[KnowledgeBaseEntry],
        distance_config: DistanceConfig,
    ) -> List[CandidateCorrection]:
        tokens = tokenize(text)
        results: List[CandidateCorrection] = []
        by_language = {}
        for entry in kb_entries:
            by_language.setdefault(entry.language, []).append(entry)

        for window_size in self.config.window_sizes:
            for start, end, substring in self.generate_windows(tokens, window_size):
                language = self.language_for(substring)
                entries = by_language.get(language, [])
                for entry in entries:
                    breakdown = self.distance_calculator.distance(
                        substring, entry.surface, language, distance_config
                    )
                    score = breakdown.total * self.config.tradeoff_lambda
                    if score <= self.config.distance_threshold:
                        results.append(CandidateCorrection(substring, entry, breakdown))
        results.sort(key=lambda c: c.distance.total)
        return results[: self.config.max_candidates]
