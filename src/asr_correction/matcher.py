"""Match ASR hypotheses against a knowledge base using phonetic distances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .config import DistanceConfig
from .distance import DistanceBreakdown, DistanceCalculator
from .phonetics import IPAConverter


_CJK_BLOCKS = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)


@dataclass(slots=True)
class KnowledgeBaseEntry:
    original: str
    normalized: str


@dataclass(slots=True)
class MatchResult:
    window: str
    start: int
    end: int
    entry: KnowledgeBaseEntry
    breakdown: DistanceBreakdown


@dataclass(slots=True)
class _Token:
    text: str
    is_cjk: bool
    start: int
    end: int


class KnowledgeBaseMatcher:
    """Sliding-window matcher over ASR hypotheses."""

    def __init__(
        self,
        knowledge_base: Sequence[str],
        *,
        config: Optional[DistanceConfig] = None,
        converter: Optional[IPAConverter] = None,
    ) -> None:
        self.config = config or DistanceConfig()
        self.converter = converter or IPAConverter()
        self.calculator = DistanceCalculator(self.config, self.converter)
        self.entries: List[KnowledgeBaseEntry] = [
            KnowledgeBaseEntry(original=item, normalized=_normalize_entry(item))
            for item in knowledge_base
        ]

    def find_matches(
        self,
        transcription: str,
        *,
        threshold: Optional[float] = None,
        min_window_size: Optional[int] = None,
        max_window_size: Optional[int] = None,
    ) -> List[MatchResult]:
        """Return candidate corrections sorted by ascending distance."""

        tokens = list(_tokenize(transcription))
        if not tokens:
            return []
        threshold = threshold if threshold is not None else self.config.threshold
        min_size = max(1, min_window_size or self.config.min_window_size)
        max_size = max(min_size, max_window_size or self.config.max_window_size)
        results: List[MatchResult] = []
        for start in range(len(tokens)):
            for size in range(min_size, max_size + 1):
                end = start + size
                if end > len(tokens):
                    break
                span_text = transcription[tokens[start].start : tokens[end - 1].end].strip()
                if not span_text:
                    continue
                for entry in self.entries:
                    breakdown = self.calculator.distance(span_text, entry.normalized)
                    if breakdown.overall <= threshold:
                        results.append(
                            MatchResult(
                                window=span_text,
                                start=tokens[start].start,
                                end=tokens[end - 1].end,
                                entry=entry,
                                breakdown=breakdown,
                            )
                        )
        results.sort(key=lambda match: match.breakdown.overall)
        return results


def _normalize_entry(entry: str) -> str:
    letters = [char for char in entry if char.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        return " ".join(char for char in entry if char.isalpha())
    return entry


def _is_cjk(char: str) -> bool:
    code = ord(char)
    for start, end in _CJK_BLOCKS:
        if start <= code <= end:
            return True
    return False


def _tokenize(text: str) -> Iterable[_Token]:
    tokens: List[_Token] = []
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if char.isspace():
            idx += 1
            continue
        start = idx
        if _is_cjk(char):
            idx += 1
            tokens.append(_Token(char, True, start, idx))
            continue
        while idx < length and not text[idx].isspace() and not _is_cjk(text[idx]):
            idx += 1
        tokens.append(_Token(text[start:idx], False, start, idx))
    return tokens


__all__ = ["KnowledgeBaseMatcher", "MatchResult"]
