"""Knowledge base utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import KnowledgeBaseEntry


@dataclass
class KnowledgeBase:
    """Simple container for known entities and jargons."""

    entries: List[KnowledgeBaseEntry]

    @classmethod
    def from_strings(
        cls, entries: Sequence[str], language: str
    ) -> "KnowledgeBase":
        return cls([KnowledgeBaseEntry(surface=entry, language=language) for entry in entries])

    def by_language(self, language: str) -> List[KnowledgeBaseEntry]:
        return [entry for entry in self.entries if entry.language == language]
