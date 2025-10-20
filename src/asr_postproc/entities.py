from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from pypinyin import Style, pinyin


def text_to_ipa_sequence(text: str) -> List[str]:
    """Return the sequence of IPA syllables for a given Chinese string.

    Characters that cannot be converted are kept as-is so that downstream
    consumers can still reason about the surface form.  The conversion relies
    on :mod:`pypinyin` with ``Style.IPA`` and does not perform segmentation.
    """

    if not text:
        return []

    ipa_syllables = pinyin(
        text,
        style=Style.IPA,
        strict=False,
        errors=lambda chars: list(chars),
    )
    flattened: List[str] = []
    for syllables in ipa_syllables:
        for syllable in syllables:
            if syllable:
                flattened.append(syllable)
    return flattened


@dataclass
class EntitySpec:
    """Specification of a canonical entity surface form.

    Parameters
    ----------
    canonical
        The canonical surface form to use in corrections.
    metadata
        Optional arbitrary metadata to attach to the entity.  This can be used
        to hold identifiers or other structured information.
    """

    canonical: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical:
            raise ValueError("Entity canonical surface form must be non-empty")
        self.metadata = dict(self.metadata)
        self._canonical_ipa: List[str] = text_to_ipa_sequence(self.canonical)

    @property
    def canonical_ipa(self) -> Sequence[str]:
        return self._canonical_ipa

    @property
    def length(self) -> int:
        return len(self.canonical)


@dataclass
class EntityMatch:
    """Represents a matched entity in an ASR transcript."""

    entity: EntitySpec
    start: int
    end: int
    similarity: float
    observed: str

    def replacement(self) -> str:
        return self.entity.canonical


def build_entity_specs(entities: Iterable[str | EntitySpec]) -> List[EntitySpec]:
    """Normalize a sequence of entity definitions into :class:`EntitySpec`.

    Strings are converted into :class:`EntitySpec` instances with no metadata.
    Existing :class:`EntitySpec` instances are returned as-is.
    """

    specs: List[EntitySpec] = []
    for entity in entities:
        if isinstance(entity, EntitySpec):
            specs.append(entity)
        else:
            specs.append(EntitySpec(canonical=str(entity)))
    return specs
