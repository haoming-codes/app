from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class KnowledgeBaseEntry:
    """Representation of a knowledge-base term."""

    canonical: str
    language: str = "zh"
    metadata: Optional[Dict[str, Any]] = None
    display: Optional[str] = None

    def __post_init__(self) -> None:
        if self.display is None:
            self.display = self.canonical


class KnowledgeBase:
    """Simple container around a list of :class:`KnowledgeBaseEntry`."""

    def __init__(self, entries: Optional[Iterable[KnowledgeBaseEntry]] = None) -> None:
        self._entries: List[KnowledgeBaseEntry] = list(entries or [])

    def add(self, entry: KnowledgeBaseEntry) -> None:
        self._entries.append(entry)

    def extend(self, entries: Iterable[KnowledgeBaseEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, item: int) -> KnowledgeBaseEntry:
        return self._entries[item]
