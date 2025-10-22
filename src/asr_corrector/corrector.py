"""Core ASR correction pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import CorrectionConfig
from .distance import DistanceBreakdown, DistanceCalculator
from .knowledge_base import KnowledgeBase, KnowledgeEntry, TOKEN_PATTERN


@dataclass
class Replacement:
    """Represents a single correction."""

    original: str
    replacement: str
    start: int
    end: int
    breakdown: DistanceBreakdown
    entry: KnowledgeEntry


@dataclass
class CorrectionResult:
    """Result of the correction process."""

    text: str
    replacements: List[Replacement]


@dataclass
class _Candidate:
    start_token: int
    end_token: int
    start: int
    end: int
    entry: KnowledgeEntry
    breakdown: DistanceBreakdown
    original: str


class ASRCorrector:
    """Correct ASR outputs using a phonetic knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase, config: CorrectionConfig) -> None:
        self.kb = knowledge_base
        self.config = config
        self.distance_calculator = DistanceCalculator(config)
        self._entries_by_length: Dict[int, List[KnowledgeEntry]] = {}
        for entry in self.kb:
            length = len(entry.tokens())
            self._entries_by_length.setdefault(length, []).append(entry)
        if self.config.window_sizes is None:
            self.window_sizes = sorted(self._entries_by_length)
        else:
            self.window_sizes = sorted(self.config.window_sizes)

    def correct(self, text: str) -> CorrectionResult:
        tokens, spans = self._tokenize_with_spans(text)
        candidates = self._generate_candidates(text, tokens, spans)
        replacements = self._select_replacements(candidates, len(tokens))
        corrected = self._apply_replacements(text, replacements)
        return CorrectionResult(text=corrected, replacements=replacements)

    def _tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        matches = list(re.finditer(TOKEN_PATTERN, text))
        tokens = [m.group(0) for m in matches]
        spans = [m.span() for m in matches]
        return tokens, spans

    def _generate_candidates(
        self,
        text: str,
        tokens: List[str],
        spans: List[Tuple[int, int]],
    ) -> List[_Candidate]:
        candidates: List[_Candidate] = []
        for start_idx in range(len(tokens)):
            for window in self.window_sizes:
                end_idx = start_idx + window
                if end_idx > len(tokens):
                    continue
                entries = self._entries_by_length.get(window, [])
                if not entries:
                    continue
                start_char = spans[start_idx][0]
                end_char = spans[end_idx - 1][1]
                original = text[start_char:end_char]
                for entry in entries:
                    breakdown = self.distance_calculator.distance(original, entry.text)
                    if breakdown.total <= self.config.threshold:
                        candidates.append(
                            _Candidate(
                                start_token=start_idx,
                                end_token=end_idx,
                                start=start_char,
                                end=end_char,
                                entry=entry,
                                breakdown=breakdown,
                                original=original,
                            )
                        )
        return candidates

    def _select_replacements(
        self, candidates: List[_Candidate], token_count: int
    ) -> List[Replacement]:
        by_start: Dict[int, List[_Candidate]] = {}
        for candidate in candidates:
            by_start.setdefault(candidate.start_token, []).append(candidate)
        occupied = set()
        replacements: List[Replacement] = []
        for start_idx in range(token_count):
            if start_idx in occupied:
                continue
            best = None
            for candidate in sorted(by_start.get(start_idx, []), key=lambda c: c.breakdown.total):
                if any(idx in occupied for idx in range(candidate.start_token, candidate.end_token)):
                    continue
                best = candidate
                break
            if best is None:
                continue
            replacements.append(
                Replacement(
                    original=best.original,
                    replacement=best.entry.text,
                    start=best.start,
                    end=best.end,
                    breakdown=best.breakdown,
                    entry=best.entry,
                )
            )
            occupied.update(range(best.start_token, best.end_token))
        replacements.sort(key=lambda r: r.start)
        return replacements

    def _apply_replacements(self, text: str, replacements: List[Replacement]) -> str:
        if not replacements:
            return text
        parts: List[str] = []
        last = 0
        for repl in replacements:
            parts.append(text[last:repl.start])
            parts.append(repl.replacement)
            last = repl.end
        parts.append(text[last:])
        return "".join(parts)
