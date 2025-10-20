"""Distance utilities for phonetic and tonal similarity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from panphon.distance import Distance as PanphonDistance

try:  # pragma: no cover - import guard
    from abydos.distance import ALINE as _ALINE
    from abydos.distance import PhoneticEditDistance as _PhoneticEditDistance
    _ABYDOS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - import guard
    _ALINE = None
    _PhoneticEditDistance = None
    _ABYDOS_IMPORT_ERROR = exc


class SegmentalDistance:
    """Base class for segmental distance metrics."""

    def distance(self, seq1: Sequence[str], seq2: Sequence[str]) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class PanphonFeatureDistance(SegmentalDistance):
    """Distance based on PanPhon's feature edit distance."""

    def __init__(self, panphon_distance: Optional[PanphonDistance] = None) -> None:
        self._distance = panphon_distance or PanphonDistance()

    def distance(self, seq1: Sequence[str], seq2: Sequence[str]) -> float:
        s1 = " ".join(seq1)
        s2 = " ".join(seq2)
        return self._distance.feature_edit_distance(s1, s2)


class AbydosPhoneticDistance(SegmentalDistance):
    """Abydos PhoneticEditDistance wrapper."""

    def __init__(self, **kwargs: float) -> None:
        if _PhoneticEditDistance is None:
            assert _ABYDOS_IMPORT_ERROR is not None
            raise ImportError("abydos is required for AbydosPhoneticDistance") from _ABYDOS_IMPORT_ERROR
        self._distance = _PhoneticEditDistance(**kwargs)

    def distance(self, seq1: Sequence[str], seq2: Sequence[str]) -> float:
        s1 = " ".join(seq1)
        s2 = " ".join(seq2)
        return self._distance.distance(s1, s2)


class AbydosALINEDistance(SegmentalDistance):
    """Abydos ALINE distance wrapper."""

    def __init__(self, **kwargs: float) -> None:
        if _ALINE is None:
            assert _ABYDOS_IMPORT_ERROR is not None
            raise ImportError("abydos is required for AbydosALINEDistance") from _ABYDOS_IMPORT_ERROR
        self._distance = _ALINE(**kwargs)

    def distance(self, seq1: Sequence[str], seq2: Sequence[str]) -> float:
        s1 = " ".join(seq1)
        s2 = " ".join(seq2)
        return self._distance.distance(s1, s2)


@dataclass
class ToneDistance:
    """Dynamic-programming tone edit distance with confusion penalties."""

    confusion_costs: Optional[Dict[Tuple[int, int], float]] = None
    substitution_default: float = 1.0
    insertion_cost: float = 1.0
    deletion_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.confusion_costs is None:
            self.confusion_costs = {
                (1, 2): 0.6,
                (2, 3): 0.5,
                (3, 4): 0.6,
                (1, 4): 0.8,
                (2, 4): 0.7,
                (1, 3): 0.7,
            }
        # Ensure symmetry in the confusion matrix.
        symmetric = {}
        for (a, b), value in self.confusion_costs.items():
            symmetric[(a, b)] = value
            symmetric[(b, a)] = value
        self.confusion_costs = symmetric

    def distance(self, tones1: Sequence[int], tones2: Sequence[int]) -> float:
        if not tones1 and not tones2:
            return 0.0
        m, n = len(tones1), len(tones2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i * self.deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = j * self.insertion_cost
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost_sub = self._substitution_cost(tones1[i - 1], tones2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + self.deletion_cost,
                    dp[i][j - 1] + self.insertion_cost,
                    dp[i - 1][j - 1] + cost_sub,
                )
        normalization = max(m, n)
        return dp[m][n] / normalization

    def _substitution_cost(self, tone_a: int, tone_b: int) -> float:
        if tone_a == tone_b:
            return 0.0
        assert self.confusion_costs is not None
        return self.confusion_costs.get((tone_a, tone_b), self.substitution_default)


@dataclass
class DistanceCalculator:
    """Combine segmental and tonal distance components."""

    segmental: SegmentalDistance
    tone: Optional[ToneDistance] = None
    segment_weight: float = 1.0
    tone_weight: float = 1.0

    def distance(self, ipa1: Sequence[str], ipa2: Sequence[str], tones1: Sequence[int], tones2: Sequence[int]) -> float:
        seg_distance = self.segmental.distance(ipa1, ipa2)
        tone_distance = 0.0
        if self.tone is not None:
            tone_distance = self.tone.distance(tones1, tones2)
        return self.segment_weight * seg_distance + self.tone_weight * tone_distance
