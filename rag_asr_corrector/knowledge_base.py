"""Knowledge base utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass
class KnowledgeBaseEntry:
    """A single pronunciation anchor."""

    text: str
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class KnowledgeBase:
    """Collection of known entities for correction."""

    entries: List[KnowledgeBaseEntry] = field(default_factory=list)

    def add(self, text: str, metadata: Optional[dict] = None) -> KnowledgeBaseEntry:
        entry = KnowledgeBaseEntry(text=text, metadata=metadata)
        self.entries.append(entry)
        return entry

    def extend(self, entries: Iterable[KnowledgeBaseEntry | str]) -> None:
        for entry in entries:
            if isinstance(entry, KnowledgeBaseEntry):
                self.entries.append(entry)
            else:
                self.entries.append(KnowledgeBaseEntry(text=entry))

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def texts(self) -> Sequence[str]:
        return [entry.text for entry in self.entries]
