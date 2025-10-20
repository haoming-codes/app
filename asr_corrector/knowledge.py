from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .config import DistanceConfig
from .distance import DistanceCalculator
from .phonetics import is_acronym, tokenize_with_spans


@dataclass
class KnowledgeEntry:
    canonical: str
    language: str

    @property
    def is_acronym(self) -> bool:
        return is_acronym(self.canonical)


@dataclass
class Correction:
    original: str
    replacement: str
    start: int
    end: int
    distance: float
    entry: KnowledgeEntry


class KnowledgeBase:
    def __init__(self, entries: Iterable[KnowledgeEntry]) -> None:
        self.entries = list(entries)

    def __iter__(self):
        return iter(self.entries)


class Corrector:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: DistanceConfig,
        window_expansion: int = 2,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.calculator = DistanceCalculator(config)
        self.window_expansion = window_expansion

    def suggest(self, text: str) -> List[Correction]:
        tokens = tokenize_with_spans(text)
        corrections: List[Correction] = []
        for entry in self.knowledge_base:
            entry_tokens = tokenize_with_spans(entry.canonical)
            token_length = max(len(entry_tokens), 1)
            max_window = token_length + self.window_expansion
            for start_idx in range(len(tokens)):
                for end_idx in range(start_idx + 1, min(len(tokens), start_idx + max_window) + 1):
                    start_pos = tokens[start_idx][1]
                    end_pos = tokens[end_idx - 1][2]
                    candidate_text = text[start_pos:end_pos]
                    distance = self.calculator.distance(
                        candidate_text,
                        entry.canonical,
                        source_language=entry.language,
                        treat_target_as_acronym=entry.is_acronym,
                    )
                    if distance <= self.calculator.config.threshold:
                        corrections.append(
                            Correction(
                                original=candidate_text,
                                replacement=entry.canonical,
                                start=start_pos,
                                end=end_pos,
                                distance=distance,
                                entry=entry,
                            )
                        )
        corrections.sort(key=lambda c: c.distance)
        return corrections

    def apply_best(self, text: str) -> str:
        corrections = self.suggest(text)
        if not corrections:
            return text
        applied: List[Correction] = []
        occupied = []
        for correction in corrections:
            if any(not (correction.end <= s or correction.start >= e) for s, e in occupied):
                continue
            applied.append(correction)
            occupied.append((correction.start, correction.end))
        if not applied:
            return text
        applied.sort(key=lambda c: c.start)
        result_parts = []
        current = 0
        for correction in applied:
            result_parts.append(text[current:correction.start])
            result_parts.append(correction.replacement)
            current = correction.end
        result_parts.append(text[current:])
        return "".join(result_parts)
