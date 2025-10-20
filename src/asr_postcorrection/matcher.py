"""Matching engine for correcting ASR named-entity errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .distance import similarity
from .ipa import ipa_sequence_to_string, text_to_ipa, text_to_ipa_sequence
from .lexicon import LexiconEntry


@dataclass(frozen=True)
class CorrectionCandidate:
    """Represents a potential correction identified in the ASR output."""

    entry: LexiconEntry
    start: int
    end: int
    score: float

    def apply(self, original: Sequence[str]) -> str:
        """Returns the corrected substring for this candidate."""
        return self.entry.surface


class CorrectionEngine:
    """Searches for near-matching substrings using IPA-based similarity."""

    def __init__(
        self,
        lexicon: Iterable[LexiconEntry],
        *,
        window_expansion: int = 1,
        threshold: float = 0.55,
    ) -> None:
        self._lexicon: List[LexiconEntry] = list(lexicon)
        if not self._lexicon:
            raise ValueError("lexicon must not be empty")
        if threshold <= 0 or threshold > 1:
            raise ValueError("threshold must be in (0, 1]")
        if window_expansion < 0:
            raise ValueError("window_expansion must be non-negative")
        self.window_expansion = window_expansion
        self.threshold = threshold
        self._lexicon_ipa = [text_to_ipa(entry.surface) for entry in self._lexicon]
        self._entry_lengths = [len(entry.surface) for entry in self._lexicon]

    def find_candidates(self, text: str) -> List[CorrectionCandidate]:
        chars = [c for c in text]
        ipa_per_char = text_to_ipa_sequence(text)
        candidates: List[CorrectionCandidate] = []

        for idx, entry in enumerate(self._lexicon):
            entry_len = self._entry_lengths[idx]
            entry_ipa = self._lexicon_ipa[idx]
            min_window = max(1, entry_len - self.window_expansion)
            max_window = entry_len + self.window_expansion

            for start in range(0, len(chars)):
                for length in range(min_window, max_window + 1):
                    end = start + length
                    if end > len(chars):
                        break
                    ipa_substring = ipa_sequence_to_string(ipa_per_char[start:end])
                    score = similarity(ipa_substring, entry_ipa)
                    if score >= self.threshold:
                        candidates.append(
                            CorrectionCandidate(entry=entry, start=start, end=end, score=score)
                        )
        return candidates

    def apply(self, text: str, candidates: Iterable[CorrectionCandidate]) -> str:
        chars = [c for c in text]
        replacements = sorted(candidates, key=lambda c: c.score, reverse=True)
        occupied = [False] * len(chars)
        accepted: List[CorrectionCandidate] = []

        for candidate in replacements:
            if any(occupied[i] for i in range(candidate.start, candidate.end)):
                continue
            accepted.append(candidate)
            for i in range(candidate.start, candidate.end):
                occupied[i] = True

        accepted.sort(key=lambda c: c.start)
        result: List[str] = []
        cursor = 0
        for candidate in accepted:
            if candidate.start > cursor:
                result.append("".join(chars[cursor : candidate.start]))
            result.append(candidate.apply(chars))
            cursor = candidate.end
        if cursor < len(chars):
            result.append("".join(chars[cursor:]))
        return "".join(result)

    def correct(self, text: str) -> str:
        candidates = self.find_candidates(text)
        return self.apply(text, candidates)

    def update_lexicon(self, entries: Iterable[LexiconEntry]) -> None:
        """Appends new entries to the lexicon."""
        for entry in entries:
            self._lexicon.append(entry)
            self._lexicon_ipa.append(text_to_ipa(entry.surface))
            self._entry_lengths.append(len(entry.surface))
