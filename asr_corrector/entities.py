"""Entity definitions for phonetic ASR correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class NameEntity:
    """Canonical representation of a named entity.

    Parameters
    ----------
    canonical
        The reference transcription (usually the correct Chinese name).
    metadata
        Optional metadata that should be carried through the correction process.
        This can store extra forms, type information, or any payload the caller
        wants to access when inspecting matches.
    ipa
        Optionally pre-computed IPA representation. When omitted the
        :class:`~asr_corrector.phonetics.PhoneticEncoder` will derive it lazily.
    """

    canonical: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ipa: Optional[str] = None

    def with_ipa(self, ipa: str) -> "NameEntity":
        """Return a copy of the entity with a computed IPA string."""

        if self.ipa == ipa:
            return self
        return NameEntity(canonical=self.canonical, metadata=dict(self.metadata), ipa=ipa)
