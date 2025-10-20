from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DistanceComponentConfig:
    """Configuration for a single distance component.

    Attributes
    ----------
    name:
        Identifier of the distance component. Supported values are ``"panphon"``,
        ``"phonetic_edit"``, ``"aline"``, ``"clts"``, and ``"tone"``.
    weight:
        Weight associated with the component when combining multiple distances.
    options:
        Arbitrary keyword arguments forwarded to the distance implementation.
    """

    name: str
    weight: float = 1.0
    options: Dict[str, object] = field(default_factory=dict)


@dataclass
class CorrectionConfig:
    """Configuration for the correction pipeline."""

    components: List[DistanceComponentConfig]
    threshold: float = 1.0
    tone_tradeoff: float = 1.0
    language_overrides: Optional[Dict[str, str]] = None

    def get_language(self, text: str, default: str) -> str:
        if not self.language_overrides:
            return default
        return self.language_overrides.get(text, default)
