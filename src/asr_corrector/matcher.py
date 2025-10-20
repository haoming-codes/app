"""Substring matching utilities for lexicon-based ASR correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

from .config import CorrectionConfig
from .distance import DistanceBreakdown, DistanceCalculator
from .phonetics import PhoneticSequence


@dataclass
class CandidateTerm:
    """Representation of a candidate lexicon entry."""

    surface: str
    metadata: Optional[Mapping[str, object]] = None


@dataclass(order=True)
class MatchResult:
    """Result of matching an ASR substring to a candidate."""

    score: float
    candidate: CandidateTerm = field(compare=False)
    start: int = field(compare=False)
    end: int = field(compare=False)
    substring: str = field(compare=False)
    segmental: float = field(compare=False)
    tone: float = field(compare=False)
    source_sequence: PhoneticSequence = field(compare=False)
    target_sequence: PhoneticSequence = field(compare=False)


class LexiconCorrector:
    """Core engine that searches for lexicon matches in ASR output."""

    def __init__(
        self,
        candidates: Iterable[CandidateTerm | str],
        config: CorrectionConfig | None = None,
        calculator: DistanceCalculator | None = None,
    ) -> None:
        self.config = config or CorrectionConfig()
        self.candidates: List[CandidateTerm] = [
            c if isinstance(c, CandidateTerm) else CandidateTerm(surface=c)
            for c in candidates
        ]
        self.calculator = calculator or DistanceCalculator(self.config.distance)

    def find_matches(self, asr_text: str) -> List[MatchResult]:
        """Return a ranked list of candidate matches within *asr_text*."""

        text_length = len(asr_text)
        results: List[MatchResult] = []
        for candidate in self.candidates:
            cand_len = max(len(candidate.surface), 1)
            min_len = max(1, cand_len - self.config.max_length_delta)
            max_len = min(text_length, cand_len + self.config.max_length_delta)
            for window in range(min_len, max_len + 1):
                for start in range(0, text_length - window + 1):
                    substring = asr_text[start : start + window]
                    breakdown = self._measure(substring, candidate.surface)
                    if breakdown.total <= self.config.threshold:
                        results.append(
                            MatchResult(
                                score=breakdown.total,
                                candidate=candidate,
                                start=start,
                                end=start + window,
                                substring=substring,
                                segmental=breakdown.segmental,
                                tone=breakdown.tone,
                                source_sequence=breakdown.source,
                                target_sequence=breakdown.target,
                            )
                        )
        results.sort()
        return results

    def _measure(self, source: str, target: str) -> DistanceBreakdown:
        return self.calculator.measure(
            source,
            target,
            normalize=self.config.enable_length_normalization,
        )
