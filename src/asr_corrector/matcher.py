"""Knowledge base matching using configurable phonetic distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import regex as reg

from .config import MatchingConfig
from .distance import DistanceCalculator, SegmentDistanceResult
from .phonetics import IPAConverter


@dataclass
class KnowledgeBaseEntry:
    """Entry in the pronunciation knowledge base."""

    surface: str
    language_hint: Optional[str] = None
    is_acronym: Optional[bool] = None
    metadata: dict | None = None
    token_length: Optional[int] = None

    def resolved_acronym(self) -> bool:
        if self.is_acronym is not None:
            return self.is_acronym
        stripped = self.surface.replace(".", "")
        return stripped.isupper()


@dataclass
class Token:
    text: str
    language: str


@dataclass
class MatchCandidate:
    entry: KnowledgeBaseEntry
    window_start: int
    window_end: int
    window_text: str
    distance: SegmentDistanceResult


class KnowledgeBaseMatcher:
    """Search ASR outputs for likely misrecognitions from a knowledge base."""

    def __init__(
        self,
        knowledge_base: Sequence[KnowledgeBaseEntry | str],
        config: Optional[MatchingConfig] = None,
        converter: Optional[IPAConverter] = None,
        distance_calculator: Optional[DistanceCalculator] = None,
    ) -> None:
        self.config = config or MatchingConfig()
        self.converter = converter or IPAConverter()
        self.distance_calculator = distance_calculator or DistanceCalculator(
            self.config.distance, converter=self.converter
        )
        self.entries = [self._ensure_entry(entry) for entry in knowledge_base]
        self._update_entry_lengths()

    def _ensure_entry(self, entry: KnowledgeBaseEntry | str) -> KnowledgeBaseEntry:
        if isinstance(entry, KnowledgeBaseEntry):
            return entry
        return KnowledgeBaseEntry(surface=entry)

    def _update_entry_lengths(self) -> None:
        for entry in self.entries:
            tokens = self._tokenize(entry.surface, language_hint=entry.language_hint)
            entry.token_length = len(tokens)

    def _tokenize(self, text: str, *, language_hint: Optional[str] = None) -> List[Token]:
        tokens: List[Token] = []
        hint = language_hint or self.config.language_hint
        for segment, lang in self.converter.segment_text(text, language_hint=hint):
            if lang.startswith('cmn'):
                tokens.extend(Token(char, lang) for char in segment if not char.isspace())
            else:
                if segment:
                    tokens.append(Token(segment, lang))
        return tokens

    def _window_to_text(self, tokens: Sequence[Token]) -> str:
        text = " ".join(token.text for token in tokens if token.text)
        text = reg.sub(r"\s+(?=\p{IsHan})", "", text)
        text = reg.sub(r"(?<=\p{IsHan})\s+", "", text)
        return text.strip()

    def match(self, asr_text: str) -> List[MatchCandidate]:
        tokens = self._tokenize(asr_text)
        results: List[MatchCandidate] = []
        if not tokens:
            return results

        for start in range(len(tokens)):
            for window_size in self.config.window_sizes:
                end = start + window_size
                partial = False
                if end > len(tokens):
                    if not self.config.allow_partial_windows:
                        break
                    end = len(tokens)
                    partial = True
                window_tokens = tokens[start:end]
                if not window_tokens:
                    continue
                window_text = self._window_to_text(window_tokens)
                for entry in self.entries:
                    if entry.token_length:
                        if len(window_tokens) == entry.token_length:
                            pass
                        elif partial and len(window_tokens) < entry.token_length:
                            pass
                        else:
                            continue
                    result = self._evaluate_window(window_text, entry)
                    if result is None:
                        continue
                    results.append(
                        MatchCandidate(
                            entry=entry,
                            window_start=start,
                            window_end=end,
                            window_text=window_text,
                            distance=result,
                        )
                    )
        results.sort(key=lambda m: m.distance.combined)
        return results

    def _evaluate_window(self, window_text: str, entry: KnowledgeBaseEntry) -> Optional[SegmentDistanceResult]:
        detail = self.distance_calculator.compute(
            window_text,
            entry.surface,
            acronym_a=False,
            acronym_b=entry.resolved_acronym(),
            language_hint_a=self.config.language_hint,
            language_hint_b=entry.language_hint,
        )
        if detail.combined > self.config.decision_threshold:
            return None
        if self.config.require_tone_match and detail.tone > 0:
            return None
        if self.config.require_stress_match and detail.stress > 0:
            return None
        return detail


__all__ = [
    "KnowledgeBaseEntry",
    "KnowledgeBaseMatcher",
    "MatchCandidate",
]
