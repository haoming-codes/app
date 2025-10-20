"""Lexicon utilities for storing named-entity pronunciations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .phonetics import text_to_ipa_segments


@dataclass
class LexiconEntry:
    """A named entity with an IPA representation."""

    surface: str
    ipa_segments: Sequence[str]
    metadata: dict | None = None

    @classmethod
    def from_text(cls, surface: str, *, metadata: dict | None = None) -> "LexiconEntry":
        return cls(surface=surface, ipa_segments=text_to_ipa_segments(surface), metadata=metadata)


class Lexicon:
    """Collection of :class:`LexiconEntry` objects."""

    def __init__(self, entries: Iterable[LexiconEntry] | None = None):
        self._entries: List[LexiconEntry] = list(entries) if entries else []

    def add(self, entry: LexiconEntry) -> None:
        self._entries.append(entry)

    def extend(self, entries: Iterable[LexiconEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Sequence[LexiconEntry]:
        return tuple(self._entries)

    @classmethod
    def from_strings(cls, surfaces: Iterable[str]) -> "Lexicon":
        return cls(LexiconEntry.from_text(surface) for surface in surfaces)


__all__ = ["Lexicon", "LexiconEntry"]
