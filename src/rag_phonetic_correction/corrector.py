"""Approximate retrieval-and-generation style corrector for Chinese ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import CorrectorConfig
from .distance import PhoneticDistanceCalculator
from .lexicon import LexiconEntry, LexiconPreparer, PreparedLexiconEntry
from .phonetics import MandarinPhonetics, SyllablePhonetics


@dataclass
class MatchCandidate:
    start: int
    end: int
    distance: float
    entry: PreparedLexiconEntry


class PhoneticRAGCorrector:
    """Apply lexicon-based phonetic post-corrections to ASR output."""

    def __init__(
        self,
        lexicon: Sequence[LexiconEntry],
        config: Optional[CorrectorConfig] = None,
        phonetics: Optional[MandarinPhonetics] = None,
    ) -> None:
        self._config = config or CorrectorConfig()
        self._phonetics = phonetics or MandarinPhonetics()
        preparer = LexiconPreparer(self._phonetics)
        self._entries: List[PreparedLexiconEntry] = [preparer.prepare(entry) for entry in lexicon]
        self._distance = PhoneticDistanceCalculator(self._config.distance)

    def correct(self, text: str) -> Tuple[str, List[MatchCandidate]]:
        syllables = self._phonetics.analyze_text(text)
        matches: List[MatchCandidate] = []
        for entry in self._entries:
            matches.extend(self._find_matches_for_entry(entry, syllables))
        matches = self._filter_matches(matches)
        corrected = self._apply_matches(text, matches)
        return corrected, matches

    # Internal helpers -------------------------------------------------

    def _find_matches_for_entry(
        self, entry: PreparedLexiconEntry, syllables: Sequence[SyllablePhonetics]
    ) -> List[MatchCandidate]:
        window = entry.syllable_count
        if window == 0:
            return []
        matches: List[MatchCandidate] = []
        for start in range(0, len(syllables) - window + 1):
            segment = syllables[start : start + window]
            if any(not s.ipa for s in segment):
                continue
            candidate_ipa = " ".join(s.ipa for s in segment)
            candidate_tones = [s.tone for s in segment]
            distance = self._distance.total_distance(
                candidate_ipa,
                entry.ipa_string,
                candidate_tones,
                entry.tones,
            )
            if distance <= self._config.threshold:
                matches.append(
                    MatchCandidate(
                        start=start,
                        end=start + window,
                        distance=distance,
                        entry=entry,
                    )
                )
        matches.sort(key=lambda m: m.distance)
        return matches[: self._config.max_matches_per_entry]

    def _filter_matches(self, matches: Sequence[MatchCandidate]) -> List[MatchCandidate]:
        ordered = sorted(matches, key=lambda m: (m.distance, m.start))
        if self._config.allow_overlaps:
            return ordered
        result: List[MatchCandidate] = []
        occupied: List[Tuple[int, int]] = []
        for match in ordered:
            if any(not (match.end <= s or match.start >= e) for s, e in occupied):
                continue
            occupied.append((match.start, match.end))
            result.append(match)
        result.sort(key=lambda m: m.start)
        return result

    def _apply_matches(self, text: str, matches: Sequence[MatchCandidate]) -> str:
        chars = list(text)
        offset = 0
        for match in matches:
            start = match.start + offset
            end = match.end + offset
            replacement = match.entry.entry.term
            chars[start:end] = list(replacement)
            offset += len(replacement) - (end - start)
        return "".join(chars)
