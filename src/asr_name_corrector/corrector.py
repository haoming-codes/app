"""Entity correction based on phonetic similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .lexicon import EntityEntry, EntityLexicon
from .phonetics import MandarinTranscriber, Transcription


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if a == b:
        return 0.0
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    distance = dp[len_a][len_b]
    return distance / max(len_a, len_b)


@dataclass
class Replacement:
    start: int
    end: int
    original: str
    entry: EntityEntry
    score: float
    phonetic_distance: float
    orthographic_distance: float

    @property
    def replacement(self) -> str:
        return self.entry.canonical or self.entry.surface


@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    replacements: List[Replacement]

    def __bool__(self) -> bool:
        return bool(self.replacements)


class EntityCorrector:
    """Correct ASR output using a lexicon and phonetic distance."""

    def __init__(
        self,
        lexicon: EntityLexicon,
        threshold: float = 0.45,
        lambda_weight: float = 0.15,
        length_slack: int = 0,
        transcriber: Optional[MandarinTranscriber] = None,
    ) -> None:
        self.lexicon = lexicon
        self.transcriber = transcriber or lexicon.transcriber
        self.threshold = threshold
        self.lambda_weight = lambda_weight
        self.length_slack = length_slack
        self.max_window = self.lexicon.max_length + self.length_slack

    def correct(self, text: str) -> CorrectionResult:
        if not text:
            return CorrectionResult(text, text, [])

        i = 0
        corrected_parts: List[str] = []
        replacements: List[Replacement] = []
        while i < len(text):
            best = self._best_replacement(text, i)
            if best and best.replacement != best.original:
                corrected_parts.append(best.replacement)
                replacements.append(best)
                i = best.end
            else:
                corrected_parts.append(text[i])
                i += 1
        corrected_text = "".join(corrected_parts)
        return CorrectionResult(text, corrected_text, replacements)

    def _best_replacement(self, text: str, index: int) -> Optional[Replacement]:
        best: Optional[Replacement] = None
        max_window = min(self.max_window, len(text) - index)
        for span in range(1, max_window + 1):
            segment = text[index : index + span]
            transcription = self.transcriber.transcribe(segment)
            for entry in self._candidate_entries(span):
                score, phonetic_distance, orthographic_distance = self._score(segment, transcription, entry)
                if score > self.threshold:
                    continue
                if best is None or score < best.score:
                    best = Replacement(
                        start=index,
                        end=index + span,
                        original=segment,
                        entry=entry,
                        score=score,
                        phonetic_distance=phonetic_distance,
                        orthographic_distance=orthographic_distance,
                    )
        return best

    def _candidate_entries(self, span: int) -> Iterable[EntityEntry]:
        lengths = {span}
        for offset in range(1, self.length_slack + 1):
            lengths.add(span + offset)
            if span - offset > 0:
                lengths.add(span - offset)
        for length in sorted(lengths):
            for entry in self.lexicon.candidates_for_length(length):
                yield entry

    def _score(
        self, segment: str, transcription: Transcription, entry: EntityEntry
    ) -> tuple[float, float, float]:
        if not entry.transcription:
            entry.transcription = self.transcriber.transcribe(entry.surface)
        entry_ipa = entry.transcription.ipa
        phonetic_distance = self.transcriber.normalized_distance(transcription.ipa, entry_ipa)
        orthographic_distance = normalized_levenshtein(segment, entry.surface)
        score = phonetic_distance + self.lambda_weight * orthographic_distance
        return score, phonetic_distance, orthographic_distance
