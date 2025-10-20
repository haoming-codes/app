"""Lexicon utilities for surface forms and canonical entities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

from .phonetics import PhoneticTranscriber


@dataclass
class SurfaceForm:
    text: str
    ipa: str


@dataclass
class LexiconEntry:
    canonical: str
    aliases: Sequence[str] = field(default_factory=tuple)
    metadata: Optional[Mapping[str, object]] = None
    surfaces: List[SurfaceForm] = field(default_factory=list)

    def all_texts(self) -> Iterable[str]:
        yield self.canonical
        for alias in self.aliases:
            yield alias


class NameLexicon:
    """Collection of named entities with phonetic projections."""

    def __init__(self, entries: Sequence[LexiconEntry], transcriber: Optional[PhoneticTranscriber] = None) -> None:
        self._entries = list(entries)
        self._transcriber = transcriber or PhoneticTranscriber()
        self._flattened: List[tuple[LexiconEntry, SurfaceForm]] = []
        self._min_len = 1
        self._max_len = 1
        self._prepare()

    def _prepare(self) -> None:
        flattened: List[tuple[LexiconEntry, SurfaceForm]] = []
        lengths = []
        for entry in self._entries:
            surfaces: List[SurfaceForm] = []
            for text in entry.all_texts():
                ipa = self._transcriber.transcribe(text)
                surface = SurfaceForm(text=text, ipa=ipa)
                surfaces.append(surface)
                flattened.append((entry, surface))
                lengths.append(len(text))
            entry.surfaces = surfaces
        if lengths:
            self._min_len = min(lengths)
            self._max_len = max(lengths)
        self._flattened = flattened

    @property
    def entries(self) -> Sequence[LexiconEntry]:
        return self._entries

    @property
    def surfaces(self) -> Sequence[tuple[LexiconEntry, SurfaceForm]]:
        return self._flattened

    @property
    def min_length(self) -> int:
        return self._min_len

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def transcriber(self) -> PhoneticTranscriber:
        return self._transcriber

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, object]],
        transcriber: Optional[PhoneticTranscriber] = None,
    ) -> "NameLexicon":
        entries = []
        for record in records:
            canonical = str(record["canonical"])
            aliases = tuple(str(alias) for alias in record.get("aliases", []) or [])
            metadata = record.get("metadata")
            entry = LexiconEntry(canonical=canonical, aliases=aliases, metadata=metadata)
            entries.append(entry)
        return cls(entries=entries, transcriber=transcriber)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        transcriber: Optional[PhoneticTranscriber] = None,
    ) -> "NameLexicon":
        import json

        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return cls.from_records(records, transcriber=transcriber)

    def iter_entries(self) -> Iterator[LexiconEntry]:
        return iter(self._entries)

    def iter_surfaces(self) -> Iterator[tuple[LexiconEntry, SurfaceForm]]:
        return iter(self._flattened)


__all__ = ["SurfaceForm", "LexiconEntry", "NameLexicon"]
