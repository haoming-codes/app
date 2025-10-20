from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .metrics import PanphonFeatureMetric, SegmentalMetric
from .phonetics import PhoneticConverter, PhoneticRepresentation, join_segments
from .tones import ToneDistance


@dataclass(frozen=True)
class LexiconEntry:
    surface: str
    metadata: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class PreparedLexiconEntry:
    entry: LexiconEntry
    representation: PhoneticRepresentation
    length: int


@dataclass(frozen=True)
class CorrectionCandidate:
    start: int
    end: int
    observed: str
    replacement: str
    total_distance: float
    segment_distance: float
    tone_distance: float
    lexicon_entry: PreparedLexiconEntry


@dataclass
class MatcherConfig:
    segment_metric: SegmentalMetric = field(default_factory=PanphonFeatureMetric)
    tone_metric: ToneDistance = field(default_factory=ToneDistance)
    lambda_segment: float = 1.0
    lambda_tone: float = 0.6
    distance_threshold: float = 1.2


class PhoneticLexiconMatcher:
    """Find lexicon matches in transcripts based on phonetic similarity."""

    def __init__(
        self,
        lexicon: Sequence[LexiconEntry | str],
        *,
        converter: Optional[PhoneticConverter] = None,
        config: Optional[MatcherConfig] = None,
    ) -> None:
        self.converter = converter or PhoneticConverter()
        self.config = config or MatcherConfig()
        self.lexicon: List[PreparedLexiconEntry] = []
        for item in lexicon:
            if isinstance(item, str):
                entry = LexiconEntry(surface=item)
            else:
                entry = item
            representation = self.converter.convert(entry.surface)
            prepared = PreparedLexiconEntry(entry=entry, representation=representation, length=len(entry.surface))
            self.lexicon.append(prepared)

    def _evaluate_candidate(self, observed: PhoneticRepresentation, target: PreparedLexiconEntry) -> CorrectionCandidate:
        segment_metric = self.config.segment_metric
        tone_metric = self.config.tone_metric
        segment_distance = segment_metric.normalized_distance(
            join_segments(observed.ipa_detoned), join_segments(target.representation.ipa_detoned)
        )
        tone_distance = tone_metric.normalized_distance(observed.tones, target.representation.tones)
        total_distance = self.config.lambda_segment * segment_distance + self.config.lambda_tone * tone_distance
        return CorrectionCandidate(
            start=0,
            end=0,
            observed=observed.text,
            replacement=target.entry.surface,
            total_distance=total_distance,
            segment_distance=segment_distance,
            tone_distance=tone_distance,
            lexicon_entry=target,
        )

    def find_matches(self, transcript: str) -> List[CorrectionCandidate]:
        matches: List[CorrectionCandidate] = []
        for prepared in self.lexicon:
            length = prepared.length
            if length == 0 or length > len(transcript):
                continue
            for start in range(0, len(transcript) - length + 1):
                observed_text = transcript[start : start + length]
                observed_repr = self.converter.convert(observed_text)
                candidate = self._evaluate_candidate(observed_repr, prepared)
                candidate = CorrectionCandidate(
                    start=start,
                    end=start + length,
                    observed=observed_text,
                    replacement=prepared.entry.surface,
                    total_distance=candidate.total_distance,
                    segment_distance=candidate.segment_distance,
                    tone_distance=candidate.tone_distance,
                    lexicon_entry=prepared,
                )
                if candidate.total_distance <= self.config.distance_threshold:
                    matches.append(candidate)
        matches.sort(key=lambda c: (c.total_distance, c.start))
        return matches

    def apply_best_corrections(self, transcript: str, matches: Sequence[CorrectionCandidate]) -> str:
        best_by_start: Dict[int, CorrectionCandidate] = {}
        for match in matches:
            current = best_by_start.get(match.start)
            if current is None or match.total_distance < current.total_distance:
                best_by_start[match.start] = match
        output: List[str] = []
        index = 0
        while index < len(transcript):
            match = best_by_start.get(index)
            if match:
                output.append(match.replacement)
                index = match.end
            else:
                output.append(transcript[index])
                index += 1
        return "".join(output)


__all__ = [
    "LexiconEntry",
    "MatcherConfig",
    "PhoneticLexiconMatcher",
    "CorrectionCandidate",
]
