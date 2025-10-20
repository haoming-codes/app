"""Combined phonetic distance calculations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance
from pyclts import CLTS
from pyclts.transcriptionsystem import TranscriptionSystem

from .config import DEFAULT_DISTANCE_CONFIG, DistanceConfig, ToneDistanceConfig
from .transcription import TranscribedSequence, Transcriber


class SegmentDistanceStrategy:
    """Base strategy for computing segmental distance."""

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        raise NotImplementedError


class PanphonSegmentDistance(SegmentDistanceStrategy):
    """Weighted feature edit distance via PanPhon."""

    def __init__(self) -> None:
        self._distance = PanphonDistance()

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        return self._distance.weighted_feature_edit_distance(ipa_a, ipa_b)


class PhoneticEditSegmentDistance(SegmentDistanceStrategy):
    """Abydos phonetic edit distance."""

    def __init__(self) -> None:
        self._distance = PhoneticEditDistance()

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        return float(self._distance.dist(ipa_a, ipa_b))


class ALINESegmentDistance(SegmentDistanceStrategy):
    """Abydos ALINE distance."""

    def __init__(self) -> None:
        self._distance = ALINE()

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        return float(self._distance.dist(ipa_a, ipa_b))


class CLTSSegmentDistance(SegmentDistanceStrategy):
    """Distance based on CLTS feature vectors."""

    def __init__(self) -> None:
        clts = CLTS()
        self._ts = TranscriptionSystem("ipa", clts=clts)
        self._feature_keys = self._collect_feature_keys()

    def _collect_feature_keys(self) -> List[str]:
        keys = set()
        for sound in self._ts.sounds():
            keys.update(sound.feature_dict().keys())
        return sorted(keys)

    def _to_vector(self, ipa: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for token in self._ts.tokens(ipa):
            try:
                sound = self._ts[token]
            except KeyError:
                continue
            features = sound.feature_dict()
            vec = []
            for key in self._feature_keys:
                value = features.get(key)
                if value == "+":
                    vec.append(1.0)
                elif value == "-":
                    vec.append(-1.0)
                elif value in {"0", ""} or value is None:
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(value))
                    except ValueError:
                        vec.append(0.0)
            vectors.append(np.array(vec, dtype=float))
        if not vectors:
            return np.zeros(len(self._feature_keys), dtype=float)
        return np.mean(vectors, axis=0)

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        vec_a = self._to_vector(ipa_a)
        vec_b = self._to_vector(ipa_b)
        if not vec_a.any() and not vec_b.any():
            return 0.0
        return float(np.linalg.norm(vec_a - vec_b))


def _build_segment_strategy(config: DistanceConfig) -> SegmentDistanceStrategy:
    if config.segment_metric == "panphon":
        return PanphonSegmentDistance()
    if config.segment_metric == "phonetic_edit":
        return PhoneticEditSegmentDistance()
    if config.segment_metric == "aline":
        return ALINESegmentDistance()
    if config.segment_metric == "clts":
        return CLTSSegmentDistance()
    raise ValueError(f"Unknown segment metric: {config.segment_metric}")


@dataclass
class ToneDistanceStrategy:
    config: ToneDistanceConfig

    def distance(self, tones_a: Iterable[int], tones_b: Iterable[int]) -> float:
        raise NotImplementedError


class WeightedToneDistance(ToneDistanceStrategy):
    """Edit-distance style cost with confusion weights."""

    def distance(self, tones_a: Iterable[int], tones_b: Iterable[int]) -> float:
        seq_a = list(tones_a)
        seq_b = list(tones_b)
        if not seq_a and not seq_b:
            return 0.0
        if not seq_a or not seq_b:
            return float(
                (len(seq_a) or len(seq_b))
                * max(self.config.confusion.deletion_cost, self.config.confusion.insertion_cost)
            )
        len_a = len(seq_a)
        len_b = len(seq_b)
        dp = np.zeros((len_a + 1, len_b + 1), dtype=float)
        for i in range(1, len_a + 1):
            dp[i, 0] = i * self.config.confusion.deletion_cost
        for j in range(1, len_b + 1):
            dp[0, j] = j * self.config.confusion.insertion_cost
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost_sub = self.config.confusion.cost(seq_a[i - 1], seq_b[j - 1])
                dp[i, j] = min(
                    dp[i - 1, j] + self.config.confusion.deletion_cost,
                    dp[i, j - 1] + self.config.confusion.insertion_cost,
                    dp[i - 1, j - 1] + cost_sub,
                )
        distance = float(dp[len_a, len_b])
        if self.config.normalize:
            return distance / max(len_a, len_b)
        return distance


class NullToneDistance(ToneDistanceStrategy):
    def distance(self, tones_a: Iterable[int], tones_b: Iterable[int]) -> float:  # type: ignore[override]
        return 0.0


def _build_tone_strategy(config: ToneDistanceConfig) -> ToneDistanceStrategy:
    if config.strategy == "weighted":
        return WeightedToneDistance(config)
    if config.strategy == "none":
        return NullToneDistance(config)
    raise ValueError(f"Unknown tone strategy: {config.strategy}")


class PhoneticDistanceCalculator:
    """Combine segmental and tonal distances according to configuration."""

    def __init__(self, config: DistanceConfig | None = None) -> None:
        self.config = deepcopy(config) if config is not None else deepcopy(DEFAULT_DISTANCE_CONFIG)
        self._segment = _build_segment_strategy(self.config)
        self._tone = _build_tone_strategy(self.config.tone_config)
        self._transcriber = Transcriber()

    @lru_cache(maxsize=1024)
    def _transcribe(self, text: str) -> TranscribedSequence:
        return self._transcriber.transcribe(text)

    def segment_distance(self, a: str, b: str) -> float:
        rep_a = self._transcribe(a)
        rep_b = self._transcribe(b)
        return self._segment.distance(rep_a.de_toned_ipa, rep_b.de_toned_ipa)

    def tone_distance(self, a: str, b: str) -> float:
        rep_a = self._transcribe(a)
        rep_b = self._transcribe(b)
        return self._tone.distance(rep_a.tones, rep_b.tones)

    def distance(self, a: str, b: str) -> float:
        seg = self.segment_distance(a, b)
        tone = self.tone_distance(a, b)
        lam = self.config.tradeoff_lambda
        combined = lam * seg + (1.0 - lam) * tone
        return combined

    def is_candidate(self, distance: float) -> bool:
        return distance <= self.config.threshold
