"""Configuration classes for ASR correction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class SegmentalMetricConfig:
    """Configuration for a single segmental distance metric.

    Attributes
    ----------
    name:
        Identifier of the metric. Supported values are
        ``"panphon_wfed"``, ``"abydos_phonetic_edit"``, ``"abydos_aline"``,
        and ``"clts"``.
    weight:
        Weight of the metric in the weighted average.
    parameters:
        Optional parameters passed to the metric implementation. Each metric
        validates the supplied parameters individually.
    """

    name: str
    weight: float = 1.0
    parameters: Dict[str, object] = field(default_factory=dict)


@dataclass
class ToneDistanceConfig:
    """Configuration for tone distance computation."""

    weight: float = 1.0
    confusion_penalty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default_penalty: float = 1.0


@dataclass
class DistanceConfig:
    """Configuration describing how to combine distances."""

    segmental_metrics: List[SegmentalMetricConfig] = field(default_factory=list)
    tone_config: Optional[ToneDistanceConfig] = None
    tone_tradeoff: float = 0.5


def default_distance_config() -> DistanceConfig:
    """Return a sensible default configuration for distance calculations."""

    tone_confusion = {
        "1": {"2": 0.6, "3": 0.8, "4": 0.9, "5": 1.0},
        "2": {"1": 0.6, "3": 0.5, "4": 0.8, "5": 0.9},
        "3": {"1": 0.8, "2": 0.5, "4": 0.6, "5": 0.9},
        "4": {"1": 0.9, "2": 0.8, "3": 0.6, "5": 1.0},
        "5": {"1": 1.0, "2": 0.9, "3": 0.9, "4": 1.0},
    }
    return DistanceConfig(
        segmental_metrics=[
            SegmentalMetricConfig(name="panphon_wfed", weight=1.0),
            SegmentalMetricConfig(name="abydos_phonetic_edit", weight=0.7),
            SegmentalMetricConfig(name="abydos_aline", weight=0.6),
            SegmentalMetricConfig(name="clts", weight=0.4),
        ],
        tone_config=ToneDistanceConfig(weight=1.0, confusion_penalty=tone_confusion, default_penalty=1.0),
        tone_tradeoff=0.35,
    )


@dataclass
class CorrectionConfig:
    """Top level configuration for :class:`~asr_corrector.corrector.ASRCorrector`."""

    distance: DistanceConfig = field(default_factory=default_distance_config)
    threshold: float = 0.4
    window_min_tokens: int = 1
    window_max_tokens: int = 6
    allow_overlapping_corrections: bool = False


@dataclass
class KnowledgeBaseEntry:
    """Represents an entry in the knowledge base."""

    term: str
    language: Optional[str] = None
    pronunciation: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None

    @property
    def is_acronym(self) -> bool:
        cleaned = self.term.replace(".", "").replace("-", "").replace(" ", "")
        return cleaned.isupper() and cleaned.isalpha()


@dataclass
class KnowledgeBase:
    """Simple list-backed knowledge base."""

    entries: List[KnowledgeBaseEntry] = field(default_factory=list)

    def add(self, entry: KnowledgeBaseEntry) -> None:
        self.entries.append(entry)

    def extend(self, entries: Iterable[KnowledgeBaseEntry]) -> None:
        self.entries.extend(entries)

    def __iter__(self):
        return iter(self.entries)
