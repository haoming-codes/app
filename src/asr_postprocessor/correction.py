"""High-level APIs for correcting Chinese ASR outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .matcher import CandidateMatcher, CandidateTerm, MatchResult, MatcherConfig, apply_matches


@dataclass
class CorrectionResult:
    """Result of running the correction pipeline."""

    original: str
    corrected: str
    matches: List[MatchResult]


class ChineseASRCorrector:
    """Correct ASR outputs by snapping near-miss substrings to canonical terms."""

    def __init__(
        self,
        candidates: Sequence[str] | Sequence[CandidateTerm],
        matcher_config: MatcherConfig | None = None,
    ) -> None:
        candidate_terms = [c if isinstance(c, CandidateTerm) else CandidateTerm(surface=c) for c in candidates]
        self.matcher = CandidateMatcher(candidate_terms, matcher_config)

    def correct(self, text: str) -> CorrectionResult:
        matches = self.matcher.find_matches(text)
        corrected = apply_matches(text, matches)
        return CorrectionResult(original=text, corrected=corrected, matches=matches)

    def batch_correct(self, texts: Iterable[str]) -> List[CorrectionResult]:
        return [self.correct(text) for text in texts]
