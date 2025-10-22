"""High-level matching helpers for ASR correction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import re

from .config import DistanceConfig
from .distances import DistanceResult, PhoneticDistanceCalculator
from .phonetics import MultilingualPhonemizer, PhoneticTranscription


@dataclass
class MatchCandidate:
    """Result for a single candidate match from the knowledge base."""

    candidate: str
    window_text: str
    distance: DistanceResult


class PhoneticMatcher:
    """Entry point that combines transcription and distance scoring."""

    def __init__(self, config: Optional[DistanceConfig] = None) -> None:
        self.config = config or DistanceConfig()
        self.phonemizer = MultilingualPhonemizer()
        self.calculator = PhoneticDistanceCalculator(self.config)

    def transcribe(self, text: str) -> PhoneticTranscription:
        """Return the phonetic transcription for inspection."""

        return self.phonemizer.transcribe(text)

    def distance(self, first: str, second: str) -> DistanceResult:
        """Compute the configured distance between two text snippets."""

        return self.calculator.distance(self.transcribe(first), self.transcribe(second))

    def match(self, transcript: str, knowledge_base: Sequence[str]) -> List[MatchCandidate]:
        """Score windows in ``transcript`` against entries in ``knowledge_base``."""

        tokens = _tokenize_units(transcript)
        results: List[MatchCandidate] = []
        for entry in knowledge_base:
            entry_tokens = _tokenize_units(entry)
            if not entry_tokens:
                continue
            best_candidate: Optional[MatchCandidate] = None
            window_sizes = _candidate_window_sizes(len(entry_tokens), self.config.window_expansion)
            entry_text = _reconstruct_text(entry_tokens)
            entry_transcription = self.transcribe(entry_text)
            for window_size in window_sizes:
                for window in _windows(tokens, window_size):
                    window_text = _reconstruct_text(window)
                    window_transcription = self.transcribe(window_text)
                    distance = self.calculator.distance(window_transcription, entry_transcription)
                    if distance.total_distance <= self.config.threshold:
                        candidate = MatchCandidate(entry, window_text, distance)
                        if best_candidate is None or distance.total_distance < best_candidate.distance.total_distance:
                            best_candidate = candidate
            if best_candidate is not None:
                results.append(best_candidate)
        results.sort(key=lambda match: match.distance.total_distance)
        return results


def _tokenize_units(text: str) -> List[str]:
    units: List[str] = []
    current = ""
    current_is_chinese: Optional[bool] = None
    current_is_alpha: Optional[bool] = None
    for char in text:
        if char.isspace():
            if current:
                units.append(current)
                current = ""
                current_is_chinese = None
                current_is_alpha = None
            continue
        is_chinese = _is_chinese(char)
        is_alpha = char.isalpha() and not is_chinese
        if is_chinese:
            if current:
                units.append(current)
                current = ""
                current_is_chinese = None
                current_is_alpha = None
            units.append(char)
            continue
        if current and ((current_is_chinese and not is_chinese) or (current_is_alpha and not is_alpha) or (not current_is_alpha and is_alpha)):
            units.append(current)
            current = ""
            current_is_chinese = None
            current_is_alpha = None
        current += char
        current_is_chinese = is_chinese
        current_is_alpha = is_alpha
    if current:
        units.append(current)
    return units


def _is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf"


def _candidate_window_sizes(base_size: int, expansion: int) -> List[int]:
    sizes = set()
    for delta in range(-expansion, expansion + 1):
        size = base_size + delta
        if size > 0:
            sizes.add(size)
    return sorted(sizes)


def _windows(tokens: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size > len(tokens):
        return
    for start in range(0, len(tokens) - size + 1):
        yield tokens[start : start + size]


def _reconstruct_text(tokens: Sequence[str]) -> str:
    parts: List[str] = []
    for token in tokens:
        if all(_is_chinese(char) for char in token):
            parts.append(token)
        else:
            if parts and not parts[-1].endswith(" "):
                parts.append(" ")
            parts.append(token)
    text = "".join(parts)
    if any(char.isalpha() for char in text):
        text = re.sub(r"\s+", " ", text).strip()
    return text
