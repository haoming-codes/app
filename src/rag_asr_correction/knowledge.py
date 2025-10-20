"""Knowledge base handling for the ASR corrector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .phonetics import (
    PhoneticRepresentation,
    build_phonetic_representation,
    detect_language,
    tokenize_for_language,
)


@dataclass(frozen=True)
class KnowledgeEntry:
    """Stores canonical information about an entity or jargon."""

    text: str
    representation: PhoneticRepresentation
    tokens: Sequence[str]


class KnowledgeBase:
    """Knowledge base storing canonical entries."""

    def __init__(self, entries: Iterable[str]):
        self._entries: List[KnowledgeEntry] = []
        for entry in entries:
            language = detect_language(entry)
            tokens = tokenize_for_language(entry, language)
            representation = build_phonetic_representation(entry, language)
            self._entries.append(
                KnowledgeEntry(
                    text=entry,
                    representation=representation,
                    tokens=tokens,
                )
            )

        self._max_token_length = max((len(e.tokens) for e in self._entries), default=1)

    @property
    def entries(self) -> Sequence[KnowledgeEntry]:
        """Return the knowledge entries."""

        return tuple(self._entries)

    @property
    def max_token_length(self) -> int:
        """Return the maximum number of tokens within an entry."""

        return self._max_token_length
