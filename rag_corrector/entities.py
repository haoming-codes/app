"""Entity definitions and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence
import json


@dataclass(frozen=True, slots=True)
class Entity:
    """Named entity that we want to enforce in transcripts."""

    surface: str
    """Canonical surface form of the entity in Chinese."""

    aliases: Sequence[str] = field(default_factory=tuple)
    """Optional aliases that should resolve to the same entity."""

    metadata: Mapping[str, object] | None = None
    """Optional metadata stored alongside the entity."""

    def candidates(self) -> Iterable[str]:
        """Yield surface form and aliases for matching."""

        yield self.surface
        for alias in self.aliases:
            yield alias


def load_entities(path: str | Path) -> List[Entity]:
    """Load entities from a JSON or JSONL file.

    The file must contain either:

    * A JSON array of objects with the keys ``surface`` (required), ``aliases``
      (optional list of strings) and ``metadata`` (optional object).
    * A JSON lines file where each line follows the same schema as the array.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, Mapping):
            raise ValueError("JSON entity file must be an array or JSONL stream")

    entities: list[Entity] = []
    for record in records:
        surface = record.get("surface")
        if not surface:
            raise ValueError(f"Entity record missing 'surface': {record!r}")
        aliases = tuple(record.get("aliases", []) or [])
        metadata = record.get("metadata")
        entities.append(Entity(surface=surface, aliases=aliases, metadata=metadata))
    return entities
