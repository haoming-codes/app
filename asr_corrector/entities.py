"""Entity definitions and lexicon helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .phonetics import BasePhoneticEncoder


@dataclass(frozen=True)
class Entity:
    """Represents a canonical named entity."""

    surface: str
    ipa: str
    aliases: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_surface(
        cls, surface: str, encoder: BasePhoneticEncoder, aliases: Sequence[str] | None = None
    ) -> "Entity":
        """Create an :class:`Entity` using the provided encoder."""

        return cls(surface=surface, ipa=encoder.encode(surface), aliases=tuple(aliases or ()))


class EntityLexicon:
    """A collection of :class:`Entity` objects with utilities for correction."""

    def __init__(self, entities: Iterable[Entity]):
        self._entities: List[Entity] = list(entities)
        if not self._entities:
            raise ValueError("EntityLexicon requires at least one entity")
        self._max_surface_length = max(len(entity.surface) for entity in self._entities)

    @classmethod
    def from_surfaces(
        cls, surfaces: Iterable[str], encoder: BasePhoneticEncoder
    ) -> "EntityLexicon":
        return cls(Entity.from_surface(surface, encoder) for surface in surfaces)

    @property
    def entities(self) -> Sequence[Entity]:
        return self._entities

    @property
    def max_surface_length(self) -> int:
        return self._max_surface_length
