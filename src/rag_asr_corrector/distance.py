from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from dtw import dtw

from .config import DistanceConfig
from .phonetics import PhonemizerService, Pronunciation


@dataclass
class DistanceBreakdown:
    """Detailed view of the pronunciation distance calculation."""

    total: float
    segment: float
    tone: float
    stress: float
    segment_components: Dict[str, float]


class PronunciationDistance:
    """Computes distances between bilingual strings."""

    def __init__(self, config: Optional[DistanceConfig] = None) -> None:
        if config is None:
            config = DistanceConfig()
        self.config = config.normalized()
        self.service = PhonemizerService(self.config)
        self._ped = PhoneticEditDistance()
        self._aline = ALINE()

    def pronunciation(self, text: str) -> Pronunciation:
        return self.service.phonemize(text)

    def distance(self, source: str, target: str) -> DistanceBreakdown:
        src = self.pronunciation(source)
        tgt = self.pronunciation(target)

        segment_components = self._segment_components(src, tgt)
        segment_distance = _combine_segment_components(
            segment_components, self.config.segment_weights
        )
        tone_distance = _tone_distance(src, tgt, self.config)
        stress_distance = _stress_distance(src, tgt, self.config)

        lambdas = self.config.lambdas
        total = (
            lambdas.segment * segment_distance
            + lambdas.tone * tone_distance
            + lambdas.stress * stress_distance
        )
        return DistanceBreakdown(
            total=float(total),
            segment=float(segment_distance),
            tone=float(tone_distance),
            stress=float(stress_distance),
            segment_components={k: float(v) for k, v in segment_components.items()},
        )

    def _segment_components(
        self, src: Pronunciation, tgt: Pronunciation
    ) -> Dict[str, float]:
        sanitized_src = src.sanitized
        sanitized_tgt = tgt.sanitized
        ped_value = float(self._ped.dist(sanitized_src, sanitized_tgt))
        aline_value = float(self._aline.dist(sanitized_src, sanitized_tgt))
        feature_value = _feature_dtw_distance(
            np.array(src.feature_vectors, dtype=float),
            np.array(tgt.feature_vectors, dtype=float),
            self.config.dtw_distance,
        )
        return {
            "phonetic_edit": ped_value,
            "aline": aline_value,
            "feature_dtw": feature_value,
        }


def _feature_dtw_distance(
    src: np.ndarray, tgt: np.ndarray, metric: str
) -> float:
    if len(src) == 0 and len(tgt) == 0:
        return 0.0
    if len(src) == 0 or len(tgt) == 0:
        return 1.0
    if metric == "manhattan":
        dist_method = lambda a, b: float(np.abs(a - b).sum())
    else:
        dist_method = lambda a, b: float(np.linalg.norm(a - b))
    result = dtw(src, tgt, dist_method=dist_method)
    path_len = max(len(result.index1), 1)
    return result.distance / path_len


def _combine_segment_components(components, weights) -> float:
    normalized_weights = weights.normalize()
    return (
        normalized_weights.phonetic_edit * components["phonetic_edit"]
        + normalized_weights.aline * components["aline"]
        + normalized_weights.feature * components["feature_dtw"]
    )


def _tone_distance(src: Pronunciation, tgt: Pronunciation, config: DistanceConfig) -> float:
    if not src.chinese_tones and not tgt.chinese_tones:
        return 0.0
    mismatches = 0
    comparisons = max(len(src.chinese_tones), len(tgt.chinese_tones))
    for s_tone, t_tone in zip(src.chinese_tones, tgt.chinese_tones):
        if s_tone != t_tone:
            mismatches += 1
    # penalize unmatched tones in longer sequence
    mismatches += abs(len(src.chinese_tones) - len(tgt.chinese_tones))
    if comparisons == 0:
        comparisons = 1
    return min(1.0, config.tone_penalty * mismatches / comparisons)


def _stress_distance(src: Pronunciation, tgt: Pronunciation, config: DistanceConfig) -> float:
    if not src.english_stress and not tgt.english_stress:
        return 0.0
    mismatches = 0
    comparisons = max(len(src.english_stress), len(tgt.english_stress))
    for s_stress, t_stress in zip(src.english_stress, tgt.english_stress):
        if s_stress != t_stress:
            mismatches += 1
    mismatches += abs(len(src.english_stress) - len(tgt.english_stress))
    if comparisons == 0:
        comparisons = 1
    return min(1.0, config.stress_penalty * mismatches / comparisons)
