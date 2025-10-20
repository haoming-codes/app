"""Data structures describing the named-entity lexicon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NamedEntity:
    """A named-entity that we want to recover from ASR output.

    Attributes
    ----------
    surface:
        The canonical surface form that should be inserted into the
        corrected transcription.
    ipa:
        The IPA representation of the canonical form. If not supplied the
        correction engine will attempt to generate it automatically using a
        phoneticizer.
    metadata:
        Optional auxiliary metadata. This can be used to store additional
        attributes such as entity type, source, etc.
    """

    surface: str
    ipa: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("NamedEntity.surface must be a non-empty string")
        if self.ipa is not None and not self.ipa.strip():
            raise ValueError("NamedEntity.ipa must be None or a non-empty string")
