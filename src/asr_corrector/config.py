"""Configuration dataclasses for ASR correction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricConfig:
    """Configuration for a segmental distance metric."""

    name: str
    weight: float = 1.0
    options: Dict[str, object] = field(default_factory=dict)


@dataclass
class ToneConfig:
    """Configuration for tone distance."""

    weight: float = 1.0
    confusion_costs: Optional[Dict[str, Dict[str, float]]] = None
    default_cost: float = 1.0


@dataclass
class CorrectionConfig:
    """Configuration for the ASR correction pipeline."""

    threshold: float = 1.5
    metrics: List[MetricConfig] = field(default_factory=list)
    tone: Optional[ToneConfig] = None
    segment_lambda: float = 1.0
    tone_lambda: float = 1.0
    window_sizes: Optional[List[int]] = None
