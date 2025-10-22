"""Sliding-window corrector built on top of the phonetic distance calculator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import PhoneticPipelineConfig
from .distance import PhoneticDistanceCalculator, SegmentDistanceBreakdown


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+|[^\w\s]")


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class CandidateMatch:
    """Proposed correction for a window of ASR output."""

    start: int
    end: int
    original: str
    replacement: str
    distance: SegmentDistanceBreakdown

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def confidence(self) -> float:
        return max(0.0, 1.0 - self.distance.total)


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    for match in _TOKEN_PATTERN.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            continue
        tokens.append(Token(piece, match.start(), match.end()))
    return tokens


class PhoneticCorrector:
    """Compare ASR output windows to knowledge-base entries using phonetic distance."""

    def __init__(
        self,
        knowledge_base: Sequence[str],
        *,
        config: PhoneticPipelineConfig | None = None,
        distance_calculator: PhoneticDistanceCalculator | None = None,
    ) -> None:
        self.knowledge_base = list(knowledge_base)
        self.config = config or PhoneticPipelineConfig()
        self.config.validate()
        self.distance_calculator = distance_calculator or PhoneticDistanceCalculator(
            self.config.aggregation
        )

    def correct(self, text: str) -> List[CandidateMatch]:
        tokens = _tokenize(text)
        if not tokens:
            return []
        proposals: Dict[Tuple[int, int], CandidateMatch] = {}
        for entry in self.knowledge_base:
            entry_tokens = _tokenize(entry)
            if not entry_tokens:
                continue
            min_len = max(1, len(entry_tokens) - self.config.aggregation.max_length_delta)
            max_len = len(entry_tokens) + self.config.aggregation.max_length_delta
            for start_index in range(len(tokens)):
                for window_len in range(min_len, max_len + 1):
                    end_index = start_index + window_len
                    if end_index > len(tokens):
                        break
                    window_tokens = tokens[start_index:end_index]
                    window_text = "".join(token.text for token in window_tokens)
                    breakdown = self.distance_calculator.distance(window_text, entry)
                    if breakdown.total > self.config.window_threshold:
                        continue
                    span = (window_tokens[0].start, window_tokens[-1].end)
                    candidate = CandidateMatch(
                        start=span[0],
                        end=span[1],
                        original=text[span[0] : span[1]],
                        replacement=entry,
                        distance=breakdown,
                    )
                    if span not in proposals or proposals[span].distance.total > breakdown.total:
                        proposals[span] = candidate

        ranked = sorted(proposals.values(), key=lambda cand: (cand.distance.total, cand.start))
        if self.config.max_candidates:
            ranked = ranked[: self.config.max_candidates]
        return ranked

    def best_match(self, text: str) -> CandidateMatch | None:
        matches = self.correct(text)
        return matches[0] if matches else None
