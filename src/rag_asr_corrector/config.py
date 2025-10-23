from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SegmentWeights:
    """Weights for the segment-level distance metrics."""

    phonetic_edit: float = 0.5
    aline: float = 0.5
    feature: float = 0.0

    def normalize(self) -> "SegmentWeights":
        total = self.phonetic_edit + self.aline + self.feature
        if total <= 0:
            raise ValueError("At least one segment weight must be positive")
        return SegmentWeights(
            phonetic_edit=self.phonetic_edit / total,
            aline=self.aline / total,
            feature=self.feature / total,
        )


@dataclass
class DistanceLambdas:
    """Top-level weights for combining the three distance components."""

    segment: float = 1.0
    tone: float = 1.0
    stress: float = 1.0

    def normalize(self) -> "DistanceLambdas":
        total = self.segment + self.tone + self.stress
        if total <= 0:
            raise ValueError("At least one lambda must be positive")
        return DistanceLambdas(
            segment=self.segment / total,
            tone=self.tone / total,
            stress=self.stress / total,
        )


@dataclass
class DistanceConfig:
    """Configuration for computing pronunciation distances."""

    segment_weights: SegmentWeights = field(default_factory=SegmentWeights)
    lambdas: DistanceLambdas = field(default_factory=DistanceLambdas)
    dtw_distance: Literal["euclidean", "manhattan"] = "euclidean"
    tone_penalty: float = 1.0
    stress_penalty: float = 1.0
    phonemizer_language_en: str = "en-us"
    phonemizer_language_cmn: str = "cmn"

    def normalized(self) -> "DistanceConfig":
        return DistanceConfig(
            segment_weights=self.segment_weights.normalize(),
            lambdas=self.lambdas.normalize(),
            dtw_distance=self.dtw_distance,
            tone_penalty=self.tone_penalty,
            stress_penalty=self.stress_penalty,
            phonemizer_language_en=self.phonemizer_language_en,
            phonemizer_language_cmn=self.phonemizer_language_cmn,
        )


@dataclass
class CorrectionConfig:
    """Configuration used by the ASR corrector pipeline."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    threshold: float = 0.4
    max_window_size: Optional[int] = None
