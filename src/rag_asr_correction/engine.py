"""Correction engine implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .config import CorrectionConfig
from .distance import DistanceBreakdown, DistanceComputer
from .knowledge import KnowledgeBase, KnowledgeEntry
from .phonetics import (
    PhoneticRepresentation,
    Token,
    build_phonetic_representation,
    detect_language,
    tokenize_with_spans,
)


@dataclass
class CorrectionCandidate:
    """Represents a potential correction for a window."""

    start_char: int
    end_char: int
    token_start_index: int
    token_end_index: int
    window_text: str
    entry: KnowledgeEntry
    score: float
    breakdown: DistanceBreakdown


@dataclass
class CorrectionResult:
    """Stores the result of the correction process."""

    original: str
    corrected: str
    applied: Sequence[CorrectionCandidate]


class CorrectionEngine:
    """RAG-inspired correction engine using a phonetic knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase, config: CorrectionConfig):
        self.knowledge_base = knowledge_base
        self.config = config
        self.distance_computer = DistanceComputer(config)
        self.max_window = min(config.max_window_size, knowledge_base.max_token_length)

    def _generate_candidates(self, text: str) -> List[CorrectionCandidate]:
        tokens = tokenize_with_spans(text)
        content_indices = [idx for idx, token in enumerate(tokens) if not token.is_space]
        candidates: List[CorrectionCandidate] = []
        window_cache: Dict[tuple[int, int], PhoneticRepresentation] = {}

        for start_pos, token_index in enumerate(content_indices):
            for window_len in range(1, self.max_window + 1):
                if start_pos + window_len > len(content_indices):
                    break

                start_char = tokens[content_indices[start_pos]].start
                end_char = tokens[content_indices[start_pos + window_len - 1]].end
                window = text[start_char:end_char]
                cache_key = (start_char, end_char)
                if cache_key not in window_cache:
                    language = detect_language(window)
                    representation = build_phonetic_representation(window, language)
                    window_cache[cache_key] = representation
                else:
                    representation = window_cache[cache_key]

                for entry in self.knowledge_base.entries:
                    if abs(len(entry.tokens) - window_len) > 2:
                        continue
                    breakdown = self.distance_computer.combined_distance(representation, entry.representation)
                    score = self.distance_computer.score(breakdown)
                    if score <= self.config.threshold:
                        candidates.append(
                            CorrectionCandidate(
                                start_char=start_char,
                                end_char=end_char,
                                token_start_index=start_pos,
                                token_end_index=start_pos + window_len - 1,
                                window_text=window,
                                entry=entry,
                                score=score,
                                breakdown=breakdown,
                            )
                        )
        return candidates

    def _select_candidates(
        self, content_tokens: Sequence[Token], candidates: Sequence[CorrectionCandidate]
    ) -> Dict[int, CorrectionCandidate]:
        best_by_start: Dict[int, CorrectionCandidate] = {}
        for candidate in candidates:
            existing = best_by_start.get(candidate.token_start_index)
            if existing is None or candidate.score < existing.score:
                best_by_start[candidate.token_start_index] = candidate
        return best_by_start

    def correct(self, text: str) -> CorrectionResult:
        tokens = tokenize_with_spans(text)
        content_indices = [idx for idx, token in enumerate(tokens) if not token.is_space]
        content_tokens = [tokens[idx] for idx in content_indices]

        if not content_tokens:
            return CorrectionResult(original=text, corrected=text, applied=[])

        candidates = self._generate_candidates(text)
        best_by_start = self._select_candidates(content_tokens, candidates)

        output_parts: List[str] = []
        applied: List[CorrectionCandidate] = []
        current_char = 0
        pos = 0

        while pos < len(content_tokens):
            token = content_tokens[pos]
            if token.start > current_char:
                output_parts.append(text[current_char:token.start])
                current_char = token.start

            candidate = best_by_start.get(pos)
            if candidate:
                output_parts.append(candidate.entry.text)
                applied.append(candidate)
                current_char = candidate.end_char
                pos = candidate.token_end_index + 1
            else:
                output_parts.append(text[current_char:token.end])
                current_char = token.end
                pos += 1

        if current_char < len(text):
            output_parts.append(text[current_char:])

        corrected = "".join(output_parts)
        return CorrectionResult(original=text, corrected=corrected, applied=applied)
