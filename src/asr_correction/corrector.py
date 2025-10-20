"""Correction engine for ASR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .config import DistanceComputationConfig, DEFAULT_CONFIG
from .distance import compute_distance
from .phonetics import normalize_acronym, tokenize_with_spans


@dataclass
class KnowledgeBaseEntry:
    """Represents a canonical entity in the knowledge base."""

    text: str

    @property
    def normalized(self) -> str:
        return normalize_acronym(self.text)

    @property
    def token_count(self) -> int:
        return len(tokenize_with_spans(self.normalized)) or 1


@dataclass
class CorrectionResult:
    """Information about a correction suggestion."""

    original_substring: str
    replacement: str
    distance: float
    start: int
    end: int


class Corrector:
    """Suggest knowledge-based replacements for ASR outputs."""

    def __init__(
        self,
        knowledge_base: Iterable[str],
        config: DistanceComputationConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.entries = [KnowledgeBaseEntry(item) for item in knowledge_base]
        self.max_tokens = max((entry.token_count for entry in self.entries), default=1)

    def suggest(self, asr_output: str) -> List[CorrectionResult]:
        """Return correction suggestions whose distance falls under the threshold."""

        tokens = tokenize_with_spans(asr_output)
        suggestions: List[CorrectionResult] = []
        for start_idx, (token, start, _) in enumerate(tokens):
            for window_size in range(1, self.max_tokens + 1):
                end_idx = start_idx + window_size
                if end_idx > len(tokens):
                    break
                window_start = tokens[start_idx][1]
                window_end = tokens[end_idx - 1][2]
                window_text = asr_output[window_start:window_end]
                normalized_window = normalize_acronym(window_text)
                for entry in self.entries:
                    distance = compute_distance(normalized_window, entry.normalized, self.config)
                    if distance <= self.config.threshold:
                        suggestions.append(
                            CorrectionResult(
                                original_substring=window_text,
                                replacement=entry.text,
                                distance=distance,
                                start=window_start,
                                end=window_end,
                            )
                        )
        suggestions.sort(key=lambda result: result.distance)
        return suggestions

    def apply_best(self, asr_output: str) -> str:
        """Apply the best non-overlapping corrections to the ASR output."""

        suggestions = self.suggest(asr_output)
        applied: List[CorrectionResult] = []
        occupied = [False] * len(asr_output)
        for suggestion in suggestions:
            if any(occupied[i] for i in range(suggestion.start, suggestion.end)):
                continue
            for i in range(suggestion.start, suggestion.end):
                occupied[i] = True
            applied.append(suggestion)
        if not applied:
            return asr_output
        result = []
        last_idx = 0
        for suggestion in sorted(applied, key=lambda s: s.start):
            result.append(asr_output[last_idx : suggestion.start])
            result.append(suggestion.replacement)
            last_idx = suggestion.end
        result.append(asr_output[last_idx:])
        return "".join(result)
