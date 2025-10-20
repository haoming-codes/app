"""Phonetic matching logic for correcting ASR output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from panphon.distance import Distance

from .lexicon import Lexicon, LexiconEntry
from .phonetics import text_to_ipa_segments


@dataclass
class MatchCandidate:
    """Potential correction for an ASR substring."""

    start: int
    end: int
    original: str
    replacement: str
    distance: float
    normalized_distance: float
    lexicon_entry: LexiconEntry


class PhoneticMatcher:
    """Search ASR output for substrings matching lexicon entries phonetically."""

    def __init__(self, lexicon: Lexicon, *, distance_metric: Distance | None = None):
        self.lexicon = lexicon
        self.distance = distance_metric or Distance()

    def _distance(self, lhs: Sequence[str], rhs: Sequence[str]) -> float:
        lhs_string = " ".join(lhs)
        rhs_string = " ".join(rhs)
        return self.distance.weighted_feature_edit_distance(lhs_string, rhs_string)

    def find_candidates(self, text: str, *, threshold: float = 1.5) -> List[MatchCandidate]:
        """Find substrings whose pronunciations are close to lexicon entries."""

        if not text or not self.lexicon:
            return []

        candidates: List[MatchCandidate] = []
        text_length = len(text)
        for entry in self.lexicon:
            target_len = len(entry.surface)
            if target_len == 0:
                continue
            max_index = text_length - target_len + 1
            for start in range(max_index):
                end = start + target_len
                substring = text[start:end]
                substring_segments = text_to_ipa_segments(substring)
                if not substring_segments:
                    continue
                dist = self._distance(substring_segments, entry.ipa_segments)
                normalizer = max(len(substring_segments), len(entry.ipa_segments))
                normalized = dist / normalizer if normalizer else dist
                if normalized <= threshold:
                    candidates.append(
                        MatchCandidate(
                            start=start,
                            end=end,
                            original=substring,
                            replacement=entry.surface,
                            distance=dist,
                            normalized_distance=normalized,
                            lexicon_entry=entry,
                        )
                    )
        candidates.sort(key=lambda item: (item.normalized_distance, item.distance))
        return candidates

    def apply_best_corrections(
        self, text: str, *, threshold: float = 1.5
    ) -> tuple[str, List[MatchCandidate]]:
        """Apply non-overlapping corrections greedily by similarity."""

        candidates = self.find_candidates(text, threshold=threshold)
        if not candidates:
            return text, []

        chosen: List[MatchCandidate] = []
        occupied = [False] * len(text)
        for candidate in candidates:
            if any(occupied[i] for i in range(candidate.start, candidate.end)):
                continue
            chosen.append(candidate)
            for i in range(candidate.start, candidate.end):
                occupied[i] = True

        if not chosen:
            return text, []

        pieces: List[str] = []
        last_index = 0
        for candidate in sorted(chosen, key=lambda item: item.start):
            if last_index < candidate.start:
                pieces.append(text[last_index:candidate.start])
            pieces.append(candidate.replacement)
            last_index = candidate.end
        if last_index < len(text):
            pieces.append(text[last_index:])
        corrected = "".join(pieces)
        return corrected, chosen


__all__ = ["MatchCandidate", "PhoneticMatcher"]
