"""Lexicon utilities for phonetic correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .phonetics import MandarinPhonetics, SyllablePhonetics


@dataclass
class LexiconEntry:
    """User-specified lexicon entry describing the canonical surface form."""

    term: str
    pronunciation: Optional[Sequence[str]] = None
    metadata: Optional[dict] = None


@dataclass
class PreparedLexiconEntry:
    """Lexicon entry with pre-computed phonetic representation."""

    entry: LexiconEntry
    syllables: List[SyllablePhonetics] = field(default_factory=list)

    @property
    def ipa_string(self) -> str:
        return " ".join(s.ipa for s in self.syllables if s.ipa)

    @property
    def tones(self) -> List[int]:
        return [s.tone for s in self.syllables if s.tone is not None]

    @property
    def syllable_count(self) -> int:
        return len([s for s in self.syllables if s.ipa])


class LexiconPreparer:
    """Helper that converts raw lexicon entries into phonetic representations."""

    def __init__(self, phonetics: Optional[MandarinPhonetics] = None) -> None:
        self._phonetics = phonetics or MandarinPhonetics()

    def prepare(self, entry: LexiconEntry) -> PreparedLexiconEntry:
        if entry.pronunciation:
            syllables = self._phonetics.analyze_pinyin(entry.pronunciation)
        else:
            syllables = self._phonetics.analyze_text(entry.term)
        return PreparedLexiconEntry(entry=entry, syllables=syllables)
