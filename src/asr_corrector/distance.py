"""Distance computation utilities for phonetic matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import DistanceConfig, ToneDistanceConfig
from .phonetics import PhoneticConverter, PhoneticSequence


class SegmentalDistance:
    """Base class for segmental distance measures."""

    def distance(self, seq1: PhoneticSequence, seq2: PhoneticSequence) -> float:
        raise NotImplementedError


class PanphonSegmentalDistance(SegmentalDistance):
    """Distance computed with panphon's weighted feature edit distance."""

    def __init__(self) -> None:
        from panphon.distance import Distance

        self._distance = Distance()

    def distance(self, seq1: PhoneticSequence, seq2: PhoneticSequence) -> float:
        s1 = seq1.concatenate()
        s2 = seq2.concatenate()
        return float(self._distance.weighted_feature_edit_distance(s1, s2))


class PhoneticEditSegmentalDistance(SegmentalDistance):
    """Distance computed with abydos' phonetic edit distance."""

    def __init__(self, **kwargs) -> None:
        from abydos.distance import PhoneticEditDistance

        self._metric = PhoneticEditDistance(**kwargs)

    def distance(self, seq1: PhoneticSequence, seq2: PhoneticSequence) -> float:
        return float(self._metric.dist(seq1.concatenate(), seq2.concatenate()))


class ALINESegmentalDistance(SegmentalDistance):
    """Distance computed with the ALINE metric from abydos."""

    def __init__(self, **kwargs) -> None:
        from abydos.distance import ALINE

        self._metric = ALINE(**kwargs)

    def distance(self, seq1: PhoneticSequence, seq2: PhoneticSequence) -> float:
        return float(self._metric.dist(seq1.concatenate(), seq2.concatenate()))


class CLTSSegmentalDistance(SegmentalDistance):
    """Distance based on CLTS phonetic feature overlap."""

    def __init__(self, repository: str | None = None) -> None:
        from pyclts import CLTS

        try:
            self._clts = CLTS(repository)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "CLTS catalogue is not available. Configure pyclts before using the"
                " CLTS-based distance."
            ) from exc
        self._bipa = self._clts.bipa

    def distance(self, seq1: PhoneticSequence, seq2: PhoneticSequence) -> float:
        features1 = [self._feature_set(s) for s in seq1.ipa_syllables]
        features2 = [self._feature_set(s) for s in seq2.ipa_syllables]
        return self._edit_distance(features1, features2)

    def _feature_set(self, ipa: str) -> frozenset[str]:
        sound = self._bipa[ipa]
        if hasattr(sound, "featuredict"):
            items = [(k, v) for k, v in sound.featuredict.items() if v]
        elif hasattr(sound, "sounds"):
            # Clusters expose a ``sounds`` attribute with individual segments.
            items: List[tuple[str, str]] = []
            for component in sound.sounds:  # type: ignore[attr-defined]
                if hasattr(component, "featuredict"):
                    items.extend((k, v) for k, v in component.featuredict.items() if v)
        else:  # pragma: no cover - defensive fallback
            items = []
        return frozenset(f"{k}={v}" for k, v in items)

    @staticmethod
    def _edit_distance(features1: Sequence[frozenset[str]], features2: Sequence[frozenset[str]]) -> float:
        if not features1:
            return float(len(features2))
        if not features2:
            return float(len(features1))
        m, n = len(features1), len(features2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = float(i)
        for j in range(1, n + 1):
            dp[0][j] = float(j)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = CLTSSegmentalDistance._substitution_cost(features1[i - 1], features2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]

    @staticmethod
    def _substitution_cost(f1: frozenset[str], f2: frozenset[str]) -> float:
        if not f1 and not f2:
            return 0.0
        if not f1 or not f2:
            return 1.0
        intersection = len(f1 & f2)
        union = len(f1 | f2)
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)


@dataclass
class ToneDistance:
    """Tone-aware edit distance using a configurable confusion matrix."""

    config: ToneDistanceConfig

    def distance(self, tones1: Sequence[int], tones2: Sequence[int]) -> float:
        m, n = len(tones1), len(tones2)
        if m == 0:
            return float(n) * self.config.insertion_penalty
        if n == 0:
            penalty = self.config.deletion_penalty or self.config.insertion_penalty
            return float(m) * penalty
        penalty = self.config.deletion_penalty or self.config.insertion_penalty
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = float(i) * penalty
        for j in range(1, n + 1):
            dp[0][j] = float(j) * self.config.insertion_penalty
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                substitution = self._substitution_cost(tones1[i - 1], tones2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + penalty,
                    dp[i][j - 1] + self.config.insertion_penalty,
                    dp[i - 1][j - 1] + substitution,
                )
        return dp[m][n]

    def _substitution_cost(self, tone_a: int, tone_b: int) -> float:
        if tone_a == tone_b:
            return 0.0
        return float(
            self.config.confusion_matrix.get(tone_a, {}).get(
                tone_b, self.config.insertion_penalty
            )
        )


@dataclass
class DistanceBreakdown:
    """Detailed report of the distance between two strings."""

    total: float
    segmental: float
    tone: float
    source: PhoneticSequence
    target: PhoneticSequence


class DistanceCalculator:
    """High level utility computing composite phonetic distances."""

    def __init__(
        self,
        config: DistanceConfig,
        converter: PhoneticConverter | None = None,
    ) -> None:
        self.config = config
        self.converter = converter or PhoneticConverter()
        self._segmental = self._build_segmental_metric()
        self._tone_distance = ToneDistance(config.tone)

    def _build_segmental_metric(self) -> SegmentalDistance:
        metric = self.config.segmental_metric.lower()
        if metric == "panphon":
            return PanphonSegmentalDistance()
        if metric == "phonetic_edit":
            return PhoneticEditSegmentalDistance(**self.config.segmental_kwargs)
        if metric == "aline":
            return ALINESegmentalDistance(**self.config.segmental_kwargs)
        if metric == "clts":
            return CLTSSegmentalDistance(**self.config.segmental_kwargs)
        raise ValueError(f"Unknown segmental metric: {self.config.segmental_metric}")

    def measure(self, source: str, target: str, normalize: bool = False) -> DistanceBreakdown:
        seq_source = self.converter.convert(source)
        seq_target = self.converter.convert(target)
        segmental = self._segmental.distance(seq_source, seq_target)
        tone = self._tone_distance.distance(seq_source.tones, seq_target.tones)
        if normalize:
            norm = max((seq_source.phonetic_length + seq_target.phonetic_length) / 2.0, 1.0)
            segmental /= norm
            tone /= norm
        total = self.config.segmental_weight * segmental + self.config.tone_weight * tone
        return DistanceBreakdown(total, segmental, tone, seq_source, seq_target)

    def distance(self, source: str, target: str, normalize: bool = False) -> float:
        """Return the combined distance only."""

        return self.measure(source, target, normalize=normalize).total
