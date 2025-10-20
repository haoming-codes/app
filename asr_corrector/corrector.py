"""ASR post-correction orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import CorrectionConfig, DistanceMetric
from .distances import combined_distance
from .phonetics import PhoneticSequence, to_phonetic_sequence


@dataclass
class LexiconEntry:
    """A canonical surface form that should be preserved in ASR outputs."""

    surface: str
    replacement: Optional[str] = None
    custom_segments: Optional[Sequence[str]] = None
    custom_tones: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if (self.custom_segments is None) != (self.custom_tones is None):
            raise ValueError("custom_segments and custom_tones must both be provided")
        if self.replacement is None:
            self.replacement = self.surface


@dataclass
class Correction:
    original: str
    replacement: str
    start: int
    end: int
    distance: float
    metric: DistanceMetric


class ASRPostCorrector:
    """Correct Chinese ASR outputs using phonetic distance."""

    def __init__(
        self,
        lexicon: Sequence[LexiconEntry],
        config: Optional[CorrectionConfig] = None,
    ) -> None:
        if not lexicon:
            raise ValueError("Lexicon must not be empty")

        self.config = config or CorrectionConfig()
        self.lexicon = list(lexicon)
        self._lexicon_sequences: Dict[int, PhoneticSequence] = {}
        for idx, entry in enumerate(self.lexicon):
            if entry.custom_segments is not None and entry.custom_tones is not None:
                self._lexicon_sequences[idx] = PhoneticSequence(
                    list(entry.custom_segments), list(entry.custom_tones)
                )
            else:
                self._lexicon_sequences[idx] = to_phonetic_sequence(entry.surface)

        self._max_entry_length = max(len(entry.surface) for entry in self.lexicon)

    def _candidate_window_sizes(self) -> Iterable[int]:
        limit = self.config.max_window_size or self._max_entry_length
        return range(1, limit + 1)

    def _score(
        self, candidate: PhoneticSequence, lexicon_seq: PhoneticSequence
    ) -> float:
        return combined_distance(
            candidate,
            lexicon_seq,
            metric=self.config.distance_metric.value,
            segmental_weight=self.config.segmental_weight,
            lambda_tone=self.config.lambda_tone,
            tone_mismatch_penalty=self.config.tone_mismatch_penalty,
        )

    def correct(self, text: str) -> Tuple[str, List[Correction]]:
        """Return a corrected string along with detailed replacements."""

        if not text:
            return text, []

        chars = list(text)
        i = 0
        corrected: List[str] = []
        corrections: List[Correction] = []
        while i < len(chars):
            best: Optional[Tuple[float, LexiconEntry, int]] = None
            best_span_end = i + 1
            for window in self._candidate_window_sizes():
                end = i + window
                if end > len(chars):
                    break
                span_text = "".join(chars[i:end])
                candidate_seq = to_phonetic_sequence(span_text)
                for idx, entry in enumerate(self.lexicon):
                    lex_seq = self._lexicon_sequences[idx]
                    score = self._score(candidate_seq, lex_seq)
                    if score <= self.config.threshold:
                        if best is None or score < best[0]:
                            best = (score, entry, idx)
                            best_span_end = end
            if best:
                _, entry, idx = best
                original = "".join(chars[i:best_span_end])
                corrected.append(entry.replacement or entry.surface)
                corrections.append(
                    Correction(
                        original=original,
                        replacement=entry.replacement or entry.surface,
                        start=i,
                        end=best_span_end,
                        distance=best[0],
                        metric=self.config.distance_metric,
                    )
                )
                i = best_span_end
            else:
                corrected.append(chars[i])
                i += 1
        return "".join(corrected), corrections
