"""Lexicon utilities for ASR post-correction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class LexiconEntry:
    """Represents a canonical named entity entry.

    Attributes:
        surface: The canonical surface form of the named entity in Chinese characters.
        metadata: Optional arbitrary metadata associated with the entry.
    """

    surface: str
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if not self.surface:
            raise ValueError("surface must be a non-empty string")
