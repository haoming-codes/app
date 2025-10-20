from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import CorrectionConfig
from .distances import PhoneticDistanceCalculator, build_components
from .knowledge import KnowledgeBase, KnowledgeBaseEntry

_CJK_RANGE = "\u4e00-\u9fff"
_CHINESE_SEGMENT = re.compile(fr"[{_CJK_RANGE}]+")
_ENGLISH_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class CorrectionSuggestion:
    original: str
    replacement: str
    start: int
    end: int
    score: float
    entry: KnowledgeBaseEntry


class CorrectionEngine:
    def __init__(self, knowledge_base: KnowledgeBase, config: CorrectionConfig) -> None:
        self.knowledge_base = knowledge_base
        self.config = config
        self.calculator = PhoneticDistanceCalculator(
            build_components(config.components), tone_tradeoff=config.tone_tradeoff
        )

    def _iter_chinese_windows(self, text: str, length: int) -> Iterable[Tuple[str, int, int]]:
        for match in _CHINESE_SEGMENT.finditer(text):
            segment = match.group()
            for idx in range(len(segment) - length + 1):
                start = match.start() + idx
                end = start + length
                yield text[start:end], start, end

    def _tokenise_english(self, text: str) -> List[Tuple[str, int, int]]:
        return [(m.group(), m.start(), m.end()) for m in _ENGLISH_WORD.finditer(text)]

    def _iter_english_windows(self, text: str, length: int) -> Iterable[Tuple[str, int, int]]:
        tokens = self._tokenise_english(text)
        if not tokens:
            return
        for idx in range(len(tokens) - length + 1):
            start = tokens[idx][1]
            end = tokens[idx + length - 1][2]
            yield text[start:end], start, end

    def _windows_for_entry(self, text: str, entry: KnowledgeBaseEntry) -> Iterable[Tuple[str, int, int]]:
        language = entry.language.lower()
        if language in {"zh", "cmn", "mandarin"}:
            length = len(entry.canonical)
            if length == 0:
                return
            yield from self._iter_chinese_windows(text, length)
        else:
            token_length = len(entry.canonical.split())
            if token_length == 0:
                return
            yield from self._iter_english_windows(text, token_length)

    def suggest_for_entry(self, text: str, entry: KnowledgeBaseEntry) -> Optional[CorrectionSuggestion]:
        language = entry.language
        best: Optional[CorrectionSuggestion] = None
        for window, start, end in self._windows_for_entry(text, entry):
            distance = self.calculator.compute(window, entry.canonical, language)
            if distance > self.config.threshold:
                continue
            if not best or distance < best.score:
                best = CorrectionSuggestion(
                    original=window,
                    replacement=entry.display or entry.canonical,
                    start=start,
                    end=end,
                    score=distance,
                    entry=entry,
                )
        return best

    def suggest(self, text: str) -> List[CorrectionSuggestion]:
        suggestions: List[CorrectionSuggestion] = []
        for entry in self.knowledge_base:
            suggestion = self.suggest_for_entry(text, entry)
            if suggestion:
                suggestions.append(suggestion)
        suggestions.sort(key=lambda s: (s.start, s.score))
        return suggestions

    def apply(self, text: str, suggestions: Sequence[CorrectionSuggestion]) -> str:
        if not suggestions:
            return text
        pieces: List[str] = []
        cursor = 0
        for suggestion in suggestions:
            if suggestion.start < cursor:
                continue
            pieces.append(text[cursor : suggestion.start])
            pieces.append(suggestion.replacement)
            cursor = suggestion.end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def correct(self, text: str) -> Tuple[str, List[CorrectionSuggestion]]:
        suggestions = self.suggest(text)
        corrected = self.apply(text, suggestions)
        return corrected, suggestions
