"""Main correction pipeline implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import CorrectionConfig, KnowledgeBase, KnowledgeBaseEntry
from .distances import DistanceCombiner
from .phonetics import PhoneticTranscriber, TranscriptionResult


@dataclass
class CorrectionSuggestion:
    start_token: int
    end_token: int
    original_text: str
    replacement_text: str
    distance: float
    entry: KnowledgeBaseEntry

    def apply(self, tokens: List[str]) -> List[str]:
        return tokens[: self.start_token] + [self.replacement_text] + tokens[self.end_token :]


class ASRCorrector:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: Optional[CorrectionConfig] = None,
        transcriber: Optional[PhoneticTranscriber] = None,
    ):
        self.knowledge_base = knowledge_base
        self.config = config or CorrectionConfig()
        self.transcriber = transcriber or PhoneticTranscriber()
        self.distance = DistanceCombiner(self.config.distance)
        self._entry_cache: Dict[int, TranscriptionResult] = {}

    def suggest(self, text: str) -> List[CorrectionSuggestion]:
        tokens = self.transcriber.segment_tokens(text)
        suggestions: List[CorrectionSuggestion] = []
        occupied: set[int] = set()
        for start, end, window in self.transcriber.windows(
            tokens, self.config.window_min_tokens, self.config.window_max_tokens
        ):
            if not self.config.allow_overlapping_corrections:
                if any(index in occupied for index in range(start, end)):
                    continue
            window_transcription = self.transcriber.transcribe(window)
            best = self._evaluate_window(window_transcription)
            if best and best.distance <= self.config.threshold:
                suggestions.append(
                    CorrectionSuggestion(
                        start_token=start,
                        end_token=end,
                        original_text=window,
                        replacement_text=best.entry.term,
                        distance=best.distance,
                        entry=best.entry,
                    )
                )
                if not self.config.allow_overlapping_corrections:
                    occupied.update(range(start, end))
        return suggestions

    def _evaluate_window(self, window_transcription: TranscriptionResult):
        best: Optional[_EvaluationResult] = None
        for entry in self.knowledge_base:
            entry_transcription = self._entry_transcription(entry)
            distance = self.distance.distance(
                window_transcription.ipa,
                entry_transcription.ipa,
                window_transcription.tone_sequence,
                entry_transcription.tone_sequence,
            )
            if best is None or distance < best.distance:
                best = _EvaluationResult(entry=entry, distance=distance)
        return best

    def _entry_transcription(self, entry: KnowledgeBaseEntry) -> TranscriptionResult:
        entry_id = id(entry)
        if entry_id not in self._entry_cache:
            if entry.pronunciation:
                tone_seq = self.transcriber._tone_sequence(entry.term)
                transcription = TranscriptionResult(
                    text=entry.term,
                    ipa=entry.pronunciation,
                    tone_sequence=tone_seq,
                )
            else:
                transcription = self.transcriber.transcribe(
                    entry.term,
                    language=entry.language,
                    is_acronym=entry.is_acronym,
                )
            self._entry_cache[entry_id] = transcription
        return self._entry_cache[entry_id]


@dataclass
class _EvaluationResult:
    entry: KnowledgeBaseEntry
    distance: float
