"""Correction utilities built on top of the distance calculator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .config import CorrectionConfig
from .distances import DistanceBreakdown, DistanceCalculator
from .ipa import MultilingualIPAConverter
from .knowledge_base import KnowledgeBase, KnowledgeBaseEntry


@dataclass
class Token:
    text: str
    kind: str
    start: int
    end: int


@dataclass
class CorrectionSuggestion:
    start: int
    end: int
    replacement: str
    breakdown: DistanceBreakdown
    entry: KnowledgeBaseEntry


class PhoneticCorrector:
    """Suggest replacements for ASR output given a knowledge base."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: CorrectionConfig | None = None,
        converter: MultilingualIPAConverter | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.config = config or CorrectionConfig()
        self.config.validate()
        self.converter = converter or MultilingualIPAConverter()
        self.calculator = DistanceCalculator(self.config.distance, self.converter)

    def distance(self, a: str, b: str) -> DistanceBreakdown:
        return self.calculator.distance(a, b)

    def ipa(self, text: str) -> str:
        return self.converter.ipa_string(text)

    def suggestions(self, text: str) -> List[CorrectionSuggestion]:
        tokens = self._tokenize(text)
        content_indices = [idx for idx, tok in enumerate(tokens) if tok.kind in {"zh", "en"}]
        suggestions: List[CorrectionSuggestion] = []

        for entry in self.knowledge_base:
            entry_tokens = [tok for tok in self._tokenize(entry.text) if tok.kind in {"zh", "en"}]
            entry_length = len(entry_tokens)
            if entry_length == 0:
                continue
            min_len = max(1, entry_length - self.config.window_tolerance)
            max_len = entry_length + self.config.window_tolerance

            for start_pos in range(len(content_indices)):
                for window_len in range(min_len, max_len + 1):
                    if start_pos + window_len > len(content_indices):
                        break
                    idx_start = content_indices[start_pos]
                    idx_end = content_indices[start_pos + window_len - 1]
                    span = tokens[idx_start].start, tokens[idx_end].end
                    substring = text[span[0] : span[1]]
                    breakdown = self.calculator.distance(substring, entry.text)
                    if breakdown.total <= self.config.threshold:
                        suggestions.append(
                            CorrectionSuggestion(
                                start=span[0],
                                end=span[1],
                                replacement=entry.text,
                                breakdown=breakdown,
                                entry=entry,
                            )
                        )
        suggestions.sort(key=lambda s: s.breakdown.total)
        return suggestions

    def apply(self, text: str) -> Tuple[str, List[CorrectionSuggestion]]:
        suggestions = self.suggestions(text)
        if not suggestions:
            return text, []

        applied: List[CorrectionSuggestion] = []
        result_parts: List[str] = []
        cursor = 0
        for suggestion in suggestions:
            if suggestion.start < cursor:
                continue
            result_parts.append(text[cursor : suggestion.start])
            result_parts.append(suggestion.replacement)
            cursor = suggestion.end
            applied.append(suggestion)
        result_parts.append(text[cursor:])
        return "".join(result_parts), applied

    def _tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        length = len(text)
        while i < length:
            char = text[i]
            if self._is_chinese(char):
                j = i + 1
                while j < length and self._is_chinese(text[j]):
                    j += 1
                tokens.append(Token(text[i:j], "zh", i, j))
                i = j
            elif char.isalpha():
                j = i + 1
                while j < length and text[j].isalpha():
                    j += 1
                tokens.append(Token(text[i:j], "en", i, j))
                i = j
            else:
                tokens.append(Token(char, "other", i, i + 1))
                i += 1
        return tokens

    @staticmethod
    def _is_chinese(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"
