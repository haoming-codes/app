"""RAG-like correction pipeline for ASR outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .lexicon import LexiconEntry, NameLexicon, SurfaceForm
from .matcher import PhoneticMatcher
from .phonetics import PhoneticTranscriber


@dataclass
class CorrectionCandidate:
    start: int
    end: int
    original: str
    replacement: str
    similarity: float
    entry: LexiconEntry


@dataclass
class CorrectionResult:
    text: str
    applied: List[CorrectionCandidate]
    skipped: List[CorrectionCandidate]


class RagBasedCorrector:
    """Approximate retrieval-augmented corrector for Chinese ASR text."""

    def __init__(
        self,
        lexicon: NameLexicon,
        matcher: Optional[PhoneticMatcher] = None,
        transcriber: Optional[PhoneticTranscriber] = None,
        threshold: float = 0.6,
    ) -> None:
        self.lexicon = lexicon
        self.matcher = matcher or PhoneticMatcher()
        self.transcriber = transcriber or lexicon.transcriber
        self.threshold = threshold

    def _iter_substrings(self, text: str) -> Iterable[tuple[int, int, str]]:
        n = len(text)
        for start in range(n):
            for length in range(self.lexicon.min_length, self.lexicon.max_length + 1):
                end = start + length
                if end > n:
                    break
                snippet = text[start:end]
                if not snippet.strip():
                    continue
                yield start, end, snippet

    def _score_snippet(self, snippet: str, surface: SurfaceForm) -> float:
        try:
            snippet_ipa = self.transcriber.transcribe(snippet)
        except ValueError:
            return 0.0
        return self.matcher.similarity(snippet_ipa, surface.ipa)

    def _best_match_for_snippet(self, snippet: str) -> Optional[CorrectionCandidate]:
        best: Optional[CorrectionCandidate] = None
        for entry, surface in self.lexicon.surfaces:
            score = self._score_snippet(snippet, surface)
            if score < self.threshold:
                continue
            if surface.text == entry.canonical and snippet == surface.text:
                continue
            candidate = CorrectionCandidate(
                start=0,
                end=0,
                original=snippet,
                replacement=entry.canonical,
                similarity=score,
                entry=entry,
            )
            if best is None or score > best.similarity:
                best = candidate
        return best

    def _offset_candidate(self, candidate: CorrectionCandidate, start: int, end: int) -> CorrectionCandidate:
        return CorrectionCandidate(
            start=start,
            end=end,
            original=candidate.original,
            replacement=candidate.replacement,
            similarity=candidate.similarity,
            entry=candidate.entry,
        )

    def propose(self, text: str) -> List[CorrectionCandidate]:
        proposals: List[CorrectionCandidate] = []
        for start, end, snippet in self._iter_substrings(text):
            candidate = self._best_match_for_snippet(snippet)
            if candidate:
                proposals.append(self._offset_candidate(candidate, start, end))
        return proposals

    def apply(self, text: str) -> CorrectionResult:
        proposals = self.propose(text)
        proposals.sort(key=lambda c: c.similarity, reverse=True)
        consumed = [False] * len(text)
        applied: List[CorrectionCandidate] = []
        skipped: List[CorrectionCandidate] = []
        for candidate in proposals:
            if any(consumed[i] for i in range(candidate.start, candidate.end)):
                skipped.append(candidate)
                continue
            for i in range(candidate.start, candidate.end):
                consumed[i] = True
            applied.append(candidate)
        applied.sort(key=lambda c: c.start)
        corrected = []
        cursor = 0
        for candidate in applied:
            corrected.append(text[cursor:candidate.start])
            corrected.append(candidate.replacement)
            cursor = candidate.end
        corrected.append(text[cursor:])
        return CorrectionResult(text="".join(corrected), applied=applied, skipped=skipped)


__all__ = ["CorrectionCandidate", "CorrectionResult", "RagBasedCorrector"]
