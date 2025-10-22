"""Knowledge base for ASR correction."""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Iterable, List

from .phonetics import ipa_for_text

TOKEN_PATTERN = r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\w]"


def tokenize(text: str) -> List[str]:
    import re

    return re.findall(TOKEN_PATTERN, text)


@dataclass
class KnowledgeEntry:
    """A canonical entity or jargon entry."""

    text: str
    metadata: dict = field(default_factory=dict)


    def tokens(self) -> List[str]:
        clean = re.sub(r'[^A-Za-z]', '', self.text)
        if clean and clean.isupper():
            return [self.text]
        return tokenize(self.text)


class KnowledgeBase:
    """Container for knowledge entries with cached phonetics."""

    def __init__(self, entries: Iterable[KnowledgeEntry]) -> None:
        self.entries: List[KnowledgeEntry] = list(entries)
        self._ipa_cache = {entry.text: ipa_for_text(entry.text) for entry in self.entries}

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def phonetics_for(self, entry: KnowledgeEntry):
        return self._ipa_cache[entry.text]
