"""ASR correction pipeline based on phonetic distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import CorrectionConfig
from .distance import PhoneticDistanceCalculator
from .phonetics import Token, tokenize_text


@dataclass(slots=True)
class KnowledgeEntry:
    """Entry in the knowledge base."""

    surface: str
    canonical: str | None = None

    @property
    def replacement(self) -> str:
        return self.canonical or self.surface


@dataclass(slots=True)
class CorrectionCandidate:
    entry: KnowledgeEntry
    start: int
    end: int
    distance: float
    raw_distance: float


class ASRCorrector:
    """Suggest corrections for ASR output using a knowledge base."""

    def __init__(
        self,
        knowledge_base: Sequence[KnowledgeEntry],
        config: CorrectionConfig | None = None,
    ) -> None:
        self.knowledge_base = list(knowledge_base)
        self.config = config or CorrectionConfig()
        self.calculator = PhoneticDistanceCalculator(self.config.distance)

    def correct(self, text: str) -> tuple[str, List[CorrectionCandidate]]:
        tokens = tokenize_text(text)
        content_indices = [i for i, t in enumerate(tokens) if t.is_content()]
        best_for_start: dict[int, CorrectionCandidate] = {}
        for entry in self.knowledge_base:
            entry_tokens = tokenize_text(entry.surface)
            entry_content = [t for t in entry_tokens if t.is_content()]
            entry_len = len(entry_content)
            if entry_len == 0:
                continue
            entry_kinds = {t.kind for t in entry_content}
            min_len = max(1, entry_len - self.config.window_radius)
            max_len = entry_len + self.config.window_radius
            for window_len in range(min_len, max_len + 1):
                if window_len > len(content_indices):
                    break
                for start_pos in range(len(content_indices) - window_len + 1):
                    start_idx = content_indices[start_pos]
                    end_idx = content_indices[start_pos + window_len - 1]
                    window_tokens = tokens[start_idx : end_idx + 1]
                    window_text = "".join(token.text for token in window_tokens)
                    window_kinds = {t.kind for t in window_tokens if t.is_content()}
                    raw_distance = self.calculator.combined_distance(window_text, entry.surface)
                    length_diff = abs(window_len - entry_len)
                    cross_script = (entry_kinds == {"cjk"} and window_kinds == {"latin"}) or (
                        entry_kinds == {"latin"} and window_kinds == {"cjk"}
                    )
                    penalty = 1.0 + (length_diff / entry_len if entry_len else 0.0)
                    if cross_script:
                        penalty = 1.0
                    if "latin" in window_kinds and "cjk" in window_kinds and not (
                        "latin" in entry_kinds and "cjk" in entry_kinds
                    ):
                        penalty *= 1.5
                    score = raw_distance * penalty
                    if score <= self.config.distance.threshold:
                        candidate = CorrectionCandidate(
                            entry=entry,
                            start=start_idx,
                            end=end_idx,
                            distance=score,
                            raw_distance=raw_distance,
                        )
                        current = best_for_start.get(start_idx)
                        if current is None or score < current.distance:
                            best_for_start[start_idx] = candidate
        candidates = list(best_for_start.values())
        applied = self._apply_candidates(tokens, candidates)
        return "".join(token.text for token in applied), candidates

    def _apply_candidates(
        self, tokens: List[Token], candidates: List[CorrectionCandidate]
    ) -> List[Token]:
        if not candidates:
            return tokens
        ordered = sorted(candidates, key=lambda c: (c.start, c.distance))
        selected: List[CorrectionCandidate] = []
        occupied: set[int] = set()
        for candidate in ordered:
            span = set(range(candidate.start, candidate.end + 1))
            if not self.config.allow_overwrite and span & occupied:
                continue
            selected.append(candidate)
            occupied.update(span)
        result = tokens[:]
        for candidate in sorted(selected, key=lambda c: c.start, reverse=True):
            replacement_tokens = tokenize_text(candidate.entry.replacement)
            result[candidate.start : candidate.end + 1] = replacement_tokens
        return result


__all__ = [
    "ASRCorrector",
    "KnowledgeEntry",
]
