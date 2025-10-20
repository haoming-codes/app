"""Distance utilities for phonetic matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
try:
    from abydos.distance import ALINE, PhoneticEditDistance
except Exception as exc:  # pragma: no cover - optional dependency guard
    ALINE = None  # type: ignore[assignment]
    PhoneticEditDistance = None  # type: ignore[assignment]
    _ABYDOS_IMPORT_ERROR = exc
else:
    _ABYDOS_IMPORT_ERROR = None

from panphon.distance import Distance as PanphonDistance

from .transcription import Syllable


@dataclass
class SegmentalDistanceConfig:
    """Configuration for segmental distance calculations."""

    metric: str = "panphon"
    method: str = "weighted_feature_edit_distance_div_maxlen"
    panphon_feature_set: str = "spe+"
    panphon_feature_model: str = "segment"


class SegmentalDistanceCalculator:
    """Compute segmental distance between two IPA sequences."""

    def __init__(self, config: SegmentalDistanceConfig) -> None:
        self.config = config
        metric = config.metric.lower()
        if metric == "panphon":
            self._distance = PanphonDistance(
                feature_set=config.panphon_feature_set,
                feature_model=config.panphon_feature_model,
            )
            self._method_name = config.method
        elif metric == "aline":
            if ALINE is None:
                raise ImportError(
                    "ALINE distance requires abydos with numpy<2.0; original error:"
                    f" {_ABYDOS_IMPORT_ERROR}"
                )
            self._distance = ALINE()
            self._method_name = "distance"
        elif metric == "phonetic_edit":
            if PhoneticEditDistance is None:
                raise ImportError(
                    "PhoneticEditDistance requires abydos with numpy<2.0; original error:"
                    f" {_ABYDOS_IMPORT_ERROR}"
                )
            self._distance = PhoneticEditDistance()
            self._method_name = "distance"
        else:
            raise ValueError(f"Unsupported segmental metric: {config.metric}")

    def distance(self, ipa1: str, ipa2: str) -> float:
        method = getattr(self._distance, self._method_name)
        return float(method(ipa1, ipa2))


_DEFAULT_TONE_MATRIX = np.array(
    [
        [0.0, 0.6, 0.8, 0.9, 0.7],
        [0.6, 0.0, 0.5, 0.8, 0.6],
        [0.8, 0.5, 0.0, 0.6, 0.5],
        [0.9, 0.8, 0.6, 0.0, 0.7],
        [0.7, 0.6, 0.5, 0.7, 0.0],
    ],
    dtype=float,
)


@dataclass
class ToneDistanceConfig:
    """Configuration for tone distance calculations."""

    confusion_matrix: np.ndarray | Sequence[Sequence[float]] = field(
        default_factory=lambda: _DEFAULT_TONE_MATRIX.copy()
    )
    insertion_penalty: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.confusion_matrix, np.ndarray):
            self.confusion_matrix = np.array(self.confusion_matrix, dtype=float)
        if self.confusion_matrix.shape != (5, 5):
            raise ValueError("Tone confusion matrix must be 5x5 (tones 1-5).")


class ToneDistanceCalculator:
    """Compute tone distance between two tone sequences using dynamic programming."""

    def __init__(self, config: ToneDistanceConfig | None = None) -> None:
        self.config = config or ToneDistanceConfig()

    def distance(self, tones1: Sequence[int], tones2: Sequence[int]) -> float:
        if not tones1 and not tones2:
            return 0.0
        m, n = len(tones1), len(tones2)
        dp = np.zeros((m + 1, n + 1), dtype=float)
        for i in range(1, m + 1):
            dp[i, 0] = i * self.config.insertion_penalty
        for j in range(1, n + 1):
            dp[0, j] = j * self.config.insertion_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                tone1 = min(max(tones1[i - 1] - 1, 0), 4)
                tone2 = min(max(tones2[j - 1] - 1, 0), 4)
                substitution = dp[i - 1, j - 1] + self.config.confusion_matrix[tone1, tone2]
                deletion = dp[i - 1, j] + self.config.insertion_penalty
                insertion = dp[i, j - 1] + self.config.insertion_penalty
                dp[i, j] = min(substitution, deletion, insertion)

        return dp[m, n] / max(m, n)


@dataclass
class CombinedDistanceConfig:
    """Configuration that combines segmental and tonal distances."""

    segmental: SegmentalDistanceConfig = field(default_factory=SegmentalDistanceConfig)
    tone: ToneDistanceConfig = field(default_factory=ToneDistanceConfig)
    segment_weight: float = 1.0
    tone_weight: float = 0.35


class PhoneticDistanceCalculator:
    """Calculate weighted phonetic distance between two syllable sequences."""

    def __init__(self, config: CombinedDistanceConfig | None = None) -> None:
        self.config = config or CombinedDistanceConfig()
        self._segmental = SegmentalDistanceCalculator(self.config.segmental)
        self._tonal = ToneDistanceCalculator(self.config.tone)

    def distance(self, syllables1: Iterable[Syllable], syllables2: Iterable[Syllable]) -> float:
        s1: List[Syllable] = list(syllables1)
        s2: List[Syllable] = list(syllables2)
        ipa1 = "".join(s.ipa or "" for s in s1)
        ipa2 = "".join(s.ipa or "" for s in s2)
        tones1 = [s.tone or 5 for s in s1]
        tones2 = [s.tone or 5 for s in s2]

        segment_distance = self._segmental.distance(ipa1, ipa2) if ipa1 or ipa2 else 0.0
        tone_distance = self._tonal.distance(tones1, tones2) if tones1 or tones2 else 0.0
        return (
            self.config.segment_weight * segment_distance
            + self.config.tone_weight * tone_distance
        )
