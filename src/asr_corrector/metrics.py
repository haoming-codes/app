from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

try:
    from panphon.distance import Distance as PanphonDistance
except ImportError:  # pragma: no cover - dependency missing at runtime
    PanphonDistance = None  # type: ignore

try:
    from abydos.distance import ALINE, PhoneticEditDistance
except ImportError:  # pragma: no cover
    ALINE = None  # type: ignore
    PhoneticEditDistance = None  # type: ignore


class SegmentalMetric(Protocol):
    name: str

    def distance(self, ipa1: str, ipa2: str) -> float:
        ...

    def normalized_distance(self, ipa1: str, ipa2: str) -> float:
        ...


@dataclass
class PanphonFeatureMetric:
    """Wrapper around panphon feature edit distance."""

    name: str = "panphon_feature_edit"
    _distance: PanphonDistance | None = None

    def _backend(self) -> PanphonDistance:
        if PanphonDistance is None:  # pragma: no cover - dependency missing
            raise ImportError("panphon is required for PanphonFeatureMetric")
        if self._distance is None:
            self._distance = PanphonDistance()
        return self._distance

    def distance(self, ipa1: str, ipa2: str) -> float:
        backend = self._backend()
        return backend.feature_edit_distance(ipa1, ipa2)

    def normalized_distance(self, ipa1: str, ipa2: str) -> float:
        dist = self.distance(ipa1, ipa2)
        denom = max(len(ipa1), len(ipa2), 1)
        return dist / float(denom)


@dataclass
class AbydosPhoneticMetric:
    """Wrapper around :class:`abydos.distance.PhoneticEditDistance`."""

    name: str = "abydos_phonetic_edit"
    _distance: PhoneticEditDistance | None = None

    def _backend(self) -> PhoneticEditDistance:
        if PhoneticEditDistance is None:  # pragma: no cover
            raise ImportError("abydos is required for AbydosPhoneticMetric")
        if self._distance is None:
            self._distance = PhoneticEditDistance()
        return self._distance

    def distance(self, ipa1: str, ipa2: str) -> float:
        backend = self._backend()
        return backend.distance(ipa1, ipa2)

    def normalized_distance(self, ipa1: str, ipa2: str) -> float:
        dist = self.distance(ipa1, ipa2)
        denom = max(len(ipa1), len(ipa2), 1)
        return dist / float(denom)


@dataclass
class AbydosALINEMetric:
    """Wrapper around :class:`abydos.distance.ALINE`."""

    name: str = "abydos_aline"
    _distance: ALINE | None = None

    def _backend(self) -> ALINE:
        if ALINE is None:  # pragma: no cover
            raise ImportError("abydos is required for AbydosALINEMetric")
        if self._distance is None:
            self._distance = ALINE()
        return self._distance

    def distance(self, ipa1: str, ipa2: str) -> float:
        backend = self._backend()
        return backend.distance(ipa1, ipa2)

    def normalized_distance(self, ipa1: str, ipa2: str) -> float:
        dist = self.distance(ipa1, ipa2)
        denom = max(len(ipa1), len(ipa2), 1)
        return dist / float(denom)


@dataclass
class SimpleLevenshteinMetric:
    """Fallback metric using raw Levenshtein distance."""

    name: str = "levenshtein"

    def distance(self, ipa1: str, ipa2: str) -> float:
        return _levenshtein_distance(ipa1, ipa2)

    def normalized_distance(self, ipa1: str, ipa2: str) -> float:
        dist = self.distance(ipa1, ipa2)
        denom = max(len(ipa1), len(ipa2), 1)
        return dist / float(denom)


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]
