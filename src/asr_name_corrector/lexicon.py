"""Lexicon support for entity correction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .phonetics import MandarinTranscriber, Transcription


@dataclass
class EntityEntry:
    surface: str
    canonical: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    transcription: Optional[Transcription] = None
    length: int = field(init=False)

    def __post_init__(self) -> None:
        if self.canonical is None:
            self.canonical = self.surface
        self.length = len(self.surface)


class EntityLexicon:
    """Store canonical entities with their phonetic representations."""

    def __init__(self, entries: Iterable[EntityEntry], transcriber: MandarinTranscriber | None = None) -> None:
        self.transcriber = transcriber or MandarinTranscriber()
        self.entries: List[EntityEntry] = []
        self.entries_by_length: Dict[int, List[EntityEntry]] = defaultdict(list)
        for entry in entries:
            transcription = self.transcriber.transcribe(entry.surface)
            entry.transcription = transcription
            entry.length = len(entry.surface)
            self.entries.append(entry)
            self.entries_by_length[entry.length].append(entry)
        self.max_length = max(self.entries_by_length.keys(), default=0)

    @classmethod
    def from_strings(cls, surfaces: Iterable[str], transcriber: MandarinTranscriber | None = None) -> "EntityLexicon":
        return cls([EntityEntry(surface=s) for s in surfaces], transcriber=transcriber)

    def candidates_for_length(self, length: int) -> List[EntityEntry]:
        return self.entries_by_length.get(length, [])

    def __iter__(self):
        return iter(self.entries)
