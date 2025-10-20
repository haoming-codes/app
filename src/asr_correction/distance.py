"""Distance computation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance
from pyclts import CLTS

from .config import DistanceComputationConfig, SegmentalMetricConfig
from .phonetics import detone_ipa, ipa_to_segments, mandarin_tone_sequence, phonemize_text
from .tones import tone_edit_distance


@dataclass
class SegmentalMetric:
    name: str
    weight: float
    options: dict

    def compute(self, ipa_a: str, ipa_b: str) -> float:
        metric = self._get_metric()
        segments_a = ipa_to_segments(detone_ipa(ipa_a))
        segments_b = ipa_to_segments(detone_ipa(ipa_b))
        if not segments_a and not segments_b:
            return 0.0
        dist = metric(segments_a, segments_b)
        normalizer = max(len(segments_a), len(segments_b), 1)
        return dist / normalizer

    def _get_metric(self):
        if self.name == "panphon":
            distance = PanphonDistance()

            def metric(a: Sequence[str], b: Sequence[str]) -> float:
                return distance.weighted_feature_edit_distance(" ".join(a), " ".join(b))

            return metric
        if self.name == "phonetic_edit_distance":
            ped = PhoneticEditDistance(**self.options)

            def metric(a: Sequence[str], b: Sequence[str]) -> float:
                return ped.dist(" ".join(a), " ".join(b))

            return metric
        if self.name == "aline":
            aline = ALINE(**self.options)

            def metric(a: Sequence[str], b: Sequence[str]) -> float:
                return aline.dist(" ".join(a), " ".join(b))

            return metric
        if self.name == "clts":
            return _clts_metric(**self.options)
        raise ValueError(f"Unsupported metric: {self.name}")


@lru_cache(maxsize=1)
def _clts_resource() -> CLTS:
    return CLTS()


def _clts_metric(distance: str = "euclidean"):
    clts = _clts_resource()
    bipa = clts.bipa

    def _feature_value(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value == "+":
                return 1.0
            if value == "-":
                return 0.0
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    def vectorize(segments: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for segment in segments:
            try:
                sound = bipa[segment]
            except KeyError:
                continue
            if not sound:
                continue
            features = sound.featuredict()
            if not features:
                continue
            ordered = sorted(features.items())
            vector = np.array([_feature_value(value) for _, value in ordered], dtype=float)
            vectors.append(vector)
        if not vectors:
            return np.zeros((0,), dtype=float)
        return np.mean(np.stack(vectors), axis=0)

    def metric(a: Sequence[str], b: Sequence[str]) -> float:
        vec_a = vectorize(a)
        vec_b = vectorize(b)
        if vec_a.size == 0 and vec_b.size == 0:
            return 0.0
        if vec_a.size == 0 or vec_b.size == 0:
            return float(max(len(a), len(b)))
        size = min(vec_a.size, vec_b.size)
        vec_a = vec_a[:size]
        vec_b = vec_b[:size]
        if distance == "euclidean":
            return float(np.linalg.norm(vec_a - vec_b))
        if distance == "cosine":
            denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
            if denom == 0:
                return 1.0
            similarity = float(np.dot(vec_a, vec_b) / denom)
            return max(0.0, 1.0 - similarity)
        raise ValueError("Unsupported CLTS distance")

    return metric


def compute_distance(text_a: str, text_b: str, config: DistanceComputationConfig) -> float:
    """Compute the combined phonetic distance between two substrings."""

    ipa_a = phonemize_text(text_a)
    ipa_b = phonemize_text(text_b)

    segmental_total = 0.0
    total_weight = 0.0
    for metric_config in config.segmental_metrics:
        metric = SegmentalMetric(
            metric_config.name,
            metric_config.weight,
            dict(metric_config.options),
        )
        segmental_total += metric.weight * metric.compute(ipa_a, ipa_b)
        total_weight += metric.weight
    if total_weight > 0:
        segmental_score = segmental_total / total_weight
    else:
        segmental_score = 0.0

    tone_score = 0.0
    if config.tone is not None and config.tone.weight > 0:
        tones_a = mandarin_tone_sequence(text_a)
        tones_b = mandarin_tone_sequence(text_b)
        tone_score = config.tone.weight * tone_edit_distance(tones_a, tones_b, config.tone)

    lambda_ = config.tradeoff_lambda
    combined = lambda_ * segmental_score + (1.0 - lambda_) * tone_score
    return combined
