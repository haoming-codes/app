"""High-level utilities for correcting Chinese ASR transcriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .matcher import MatchCandidate, PhoneticEntityMatcher


@dataclass(frozen=True)
class CorrectionResult:
    """Represents a correction applied to the ASR output."""

    start: int
    end: int
    replacement: str
    original: str
    score: float


class ASRNamedEntityCorrector:
    """Apply phonetic named-entity corrections to an ASR transcription."""

    def __init__(self, entities: Iterable[str], *, threshold: float = 0.6) -> None:
        self._matcher = PhoneticEntityMatcher(entities, threshold=threshold)

    def correct(self, transcription: str) -> tuple[str, List[CorrectionResult]]:
        """Return the corrected transcription and the applied corrections."""

        matches = self._matcher.find_matches(transcription)
        corrected_text = self._apply_corrections(transcription, matches)
        results = [
            CorrectionResult(
                start=match.start,
                end=match.end,
                replacement=match.entity,
                original=match.original,
                score=match.score,
            )
            for match in matches
        ]
        return corrected_text, results

    @staticmethod
    def _apply_corrections(text: str, matches: List[MatchCandidate]) -> str:
        if not matches:
            return text

        parts: List[str] = []
        cursor = 0
        for match in matches:
            parts.append(text[cursor : match.start])
            parts.append(match.entity)
            cursor = match.end
        parts.append(text[cursor:])
        return "".join(parts)


__all__ = ["ASRNamedEntityCorrector", "CorrectionResult"]
