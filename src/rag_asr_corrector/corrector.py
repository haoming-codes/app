from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import CorrectionConfig
from .distance import DistanceBreakdown, PronunciationDistance
from .phonetics import BilingualToken, Pronunciation


@dataclass
class KnowledgeEntry:
    """Preprocessed entry from the knowledge base."""

    text: str
    tokens: List[BilingualToken]
    pronunciation: Pronunciation


@dataclass
class CorrectionSuggestion:
    """Potential correction for an ASR window."""

    start: int
    end: int
    original_text: str
    candidate: str
    distance: DistanceBreakdown


class ASRCorrector:
    """Suggests corrections by comparing to a knowledge base."""

    def __init__(
        self,
        knowledge_base: Sequence[str],
        config: CorrectionConfig | None = None,
    ) -> None:
        if config is None:
            config = CorrectionConfig()
        self.config = config
        self._distance = PronunciationDistance(self.config.distance)
        service = self._distance.service
        self._entries: List[KnowledgeEntry] = []
        for text in knowledge_base:
            tokens = service.tokenize(text)
            pronunciation = self._distance.pronunciation(text)
            self._entries.append(
                KnowledgeEntry(text=text, tokens=tokens, pronunciation=pronunciation)
            )

    def suggest(self, asr_text: str) -> List[CorrectionSuggestion]:
        tokens = self._distance.service.tokenize(asr_text)
        suggestions: List[CorrectionSuggestion] = []
        for entry in self._entries:
            token_count = len(entry.tokens)
            if token_count == 0:
                continue
            if self.config.max_window_size and token_count > self.config.max_window_size:
                continue
            for start in range(0, max(len(tokens) - token_count + 1, 0)):
                window = tokens[start : start + token_count]
                if not _languages_match(entry.tokens, window):
                    continue
                window_text = _slice_text(asr_text, window)
                breakdown = self._distance.distance(window_text, entry.text)
                if breakdown.total <= self.config.threshold:
                    suggestions.append(
                        CorrectionSuggestion(
                            start=window[0].start,
                            end=window[-1].end,
                            original_text=window_text,
                            candidate=entry.text,
                            distance=breakdown,
                        )
                    )
        suggestions.sort(key=lambda s: s.distance.total)
        return suggestions


def _languages_match(reference: Sequence[BilingualToken], window: Sequence[BilingualToken]) -> bool:
    if len(reference) != len(window):
        return False
    return all(r.language == w.language for r, w in zip(reference, window))


def _slice_text(text: str, tokens: Sequence[BilingualToken]) -> str:
    if not tokens:
        return ""
    start = tokens[0].start
    end = tokens[-1].end
    return text[start:end]
