from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import DistanceConfig
from .distance import DistanceBreakdown, DistanceCalculator


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class CorrectionCandidate:
    entry: str
    original: str
    start: int
    end: int
    distance: DistanceBreakdown


@dataclass
class CorrectionResult:
    text: str
    applied: List[CorrectionCandidate]


class CorrectionEngine:
    """Suggest replacements for ASR output using a knowledge base."""

    def __init__(self, knowledge_base: Sequence[str], config: DistanceConfig | None = None) -> None:
        self.knowledge_base = list(dict.fromkeys(knowledge_base))
        self.config = config or DistanceConfig()
        self.calculator = DistanceCalculator(self.config)

    def correct(self, text: str) -> CorrectionResult:
        tokens = list(_tokenize(text))
        candidates: List[CorrectionCandidate] = []
        for entry in self.knowledge_base:
            best_candidate = self._best_match_for_entry(text, tokens, entry)
            if best_candidate is not None:
                candidates.append(best_candidate)
        applied = self._select_candidates(candidates)
        corrected = _apply_replacements(text, applied)
        return CorrectionResult(text=corrected, applied=applied)

    def _best_match_for_entry(self, text: str, tokens: Sequence[Token], entry: str) -> CorrectionCandidate | None:
        entry_tokens = list(_tokenize(entry))
        if not entry_tokens:
            return None
        target_length = len(entry_tokens)
        radius = max(0, self.config.window_radius)
        lengths = range(max(1, target_length - radius), target_length + radius + 1)
        best: CorrectionCandidate | None = None
        threshold = self.config.correction_threshold
        for length in lengths:
            if length == 0:
                continue
            for start in range(0, len(tokens) - length + 1):
                window_tokens = tokens[start : start + length]
                span_start = window_tokens[0].start
                span_end = window_tokens[-1].end
                candidate_text = text[span_start:span_end]
                distance = self.calculator.compute(candidate_text, entry)
                if distance.total <= threshold:
                    if best is None or distance.total < best.distance.total:
                        best = CorrectionCandidate(
                            entry=entry,
                            original=candidate_text,
                            start=span_start,
                            end=span_end,
                            distance=distance,
                        )
        return best

    def _select_candidates(self, candidates: Iterable[CorrectionCandidate]) -> List[CorrectionCandidate]:
        sorted_candidates = sorted(candidates, key=lambda c: (c.distance.total, c.end - c.start))
        selected: List[CorrectionCandidate] = []
        occupied: List[tuple[int, int]] = []
        for cand in sorted_candidates:
            if any(not (cand.end <= s or cand.start >= e) for s, e in occupied):
                continue
            selected.append(cand)
            occupied.append((cand.start, cand.end))
        selected.sort(key=lambda c: c.start)
        return selected


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]|\d+")


def _tokenize(text: str) -> Iterable[Token]:
    for match in _TOKEN_PATTERN.finditer(text):
        yield Token(match.group(), match.start(), match.end())


def _apply_replacements(text: str, candidates: Iterable[CorrectionCandidate]) -> str:
    result = []
    last_index = 0
    for cand in candidates:
        if cand.start < last_index:
            continue
        result.append(text[last_index:cand.start])
        result.append(cand.entry)
        last_index = cand.end
    result.append(text[last_index:])
    return "".join(result)


__all__ = ["CorrectionEngine", "CorrectionResult", "CorrectionCandidate"]
