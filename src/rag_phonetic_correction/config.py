"""Configuration dataclasses for phonetic distance and correction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

ToneConfusionMatrix = Mapping[Tuple[int, int], float]


@dataclass
class PhoneticDistanceConfig:
    """Hyperparameters that control phonetic distance computation."""

    panphon_weight: float = 1.0
    aline_weight: float = 0.0
    phonetic_edit_weight: float = 0.0
    tone_weight: float = 0.3
    tone_gap_penalty: float = 1.0
    tone_default_penalty: float = 1.0
    tone_confusion: Optional[ToneConfusionMatrix] = None
    normalization_exponent: float = 1.0

    def metric_weights(self) -> Mapping[str, float]:
        """Return the weights for the available segmental metrics."""

        return {
            "panphon": self.panphon_weight,
            "aline": self.aline_weight,
            "phonetic_edit": self.phonetic_edit_weight,
        }


@dataclass
class CorrectorConfig:
    """Configuration for the phonetic RAG corrector."""

    distance: PhoneticDistanceConfig = field(default_factory=PhoneticDistanceConfig)
    threshold: float = 1.5
    max_matches_per_entry: int = 1
    allow_overlaps: bool = False

    def update_tone_confusion(self, pairs: Iterable[Tuple[Tuple[int, int], float]]) -> None:
        """Convenience helper to update the tone confusion mapping in-place."""

        if self.distance.tone_confusion is None:
            self.distance.tone_confusion = {}
        if isinstance(self.distance.tone_confusion, MutableMapping):
            for key, value in pairs:
                self.distance.tone_confusion[key] = value
        else:  # pragma: no cover - defensive, for read-only mappings
            merged: Dict[Tuple[int, int], float] = dict(self.distance.tone_confusion)
            for key, value in pairs:
                merged[key] = value
            self.distance.tone_confusion = merged
