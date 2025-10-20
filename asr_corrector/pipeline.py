"""High-level correction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import DistanceConfig, KnowledgeBaseEntry, MatcherConfig
from .distance import DistanceBreakdown
from .knowledge_base import KnowledgeBase
from .matcher import CandidateCorrection, WindowMatcher


@dataclass
class CorrectionResult:
    """Holds the proposed corrections for an ASR output."""

    original: str
    candidates: List[CandidateCorrection]


class ASRCorrector:
    """Correct ASR transcriptions by matching against a knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase, matcher_config: MatcherConfig, distance_config: DistanceConfig) -> None:
        self.knowledge_base = knowledge_base
        self.matcher_config = matcher_config
        self.distance_config = distance_config
        self.matcher = WindowMatcher(matcher_config)

    def suggest(self, asr_text: str) -> CorrectionResult:
        entries = self.knowledge_base.entries
        candidates = self.matcher.match(asr_text, entries, self.distance_config)
        return CorrectionResult(original=asr_text, candidates=candidates)

    def compute_distance(self, text_a: str, text_b: str, language: str) -> DistanceBreakdown:
        """Convenience wrapper to compute distances for hyperparameter tuning."""

        return self.matcher.distance_calculator.distance(text_a, text_b, language, self.distance_config)
