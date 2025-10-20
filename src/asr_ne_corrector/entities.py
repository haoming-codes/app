"""Data structures that represent named entities and matches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NameEntity:
    """Represents a canonical named entity.

    Attributes
    ----------
    text:
        The canonical Chinese representation of the entity.
    ipa:
        A whitespace-delimited IPA transcription of ``text``.
    """

    text: str
    ipa: str

    @property
    def length(self) -> int:
        """Number of characters in the canonical text."""

        return len(self.text)


@dataclass(frozen=True)
class EntityMatch:
    """Represents an alignment between a substring and a named entity."""

    start: int
    end: int
    observed: str
    entity: NameEntity
    score: float

    def replacement(self) -> str:
        """Return the canonical text to use as a replacement."""

        return self.entity.text
