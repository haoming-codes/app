"""Phonetic distance computation utilities."""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from clts import CLTS
from epitran import Epitran
from panphon.distance import Distance as PanphonDistance

from .config import (
    DistanceConfig,
    FeatureDistance,
    SegmentalMetric,
    SegmentalMetricConfig,
)
from .tones import extract_tones, tone_distance


@dataclass
class DistanceBreakdown:
    """Holds a breakdown of the computed distance components."""

    segmental: float
    tone: float
    total: float
    details: Dict[str, float]


class IPATranscriber:
    """Converts orthographic text into IPA strings using Epitran."""

    def __init__(self, language_map: Dict[str, str]) -> None:
        self._language_map = language_map
        self._cache: Dict[str, Epitran] = {}

    def transliterate(self, text: str, language: str) -> str:
        epi = self._cache.get(language)
        if epi is None:
            epi_code = self._language_map.get(language, language)
            epi = Epitran(epi_code)
            self._cache[language] = epi
        return epi.transliterate(text)


def strip_tone_marks(ipa: str) -> str:
    """Return the IPA string with tone diacritics removed."""

    stripped_chars: list[str] = []
    for char in ipa:
        # Keep tone letters but drop digits and combining marks in the tone ranges.
        if char.isdigit():
            continue
        cat = ord(char)
        if 0x0300 <= cat <= 0x036F:
            # Combining diacritics
            continue
        stripped_chars.append(char)
    return "".join(stripped_chars)


class SegmentalDistanceCalculator:
    """Computes a weighted sum of segmental metrics."""

    def __init__(self) -> None:
        self._panphon = PanphonDistance()
        self._aline = ALINE()
        self._phonetic_edit = PhoneticEditDistance()
        self._clts = CLTS()

    def compute(
        self, ipa_a: str, ipa_b: str, configs: Sequence[SegmentalMetricConfig]
    ) -> Tuple[float, Dict[str, float]]:
        if not configs:
            return 0.0, {}
        detail: Dict[str, float] = {}
        total = 0.0
        for config in configs:
            if config.metric == SegmentalMetric.PANPHON:
                dist = self._panphon.weighted_feature_edit_distance(ipa_a, ipa_b)
            elif config.metric == SegmentalMetric.PHONETIC_EDIT:
                dist = self._phonetic_edit.dist(ipa_a, ipa_b)
            elif config.metric == SegmentalMetric.ALINE:
                dist = self._aline.dist(ipa_a, ipa_b)
            elif config.metric == SegmentalMetric.CLTS:
                dist = self._clts_distance(ipa_a, ipa_b, config.feature_distance)
            else:  # pragma: no cover - defensive fallback
                raise ValueError(f"Unsupported metric {config.metric}")
            detail[f"segmental::{config.metric.value}"] = dist
            total += config.weight * dist
        return total, detail

    def _clts_distance(self, ipa_a: str, ipa_b: str, feature_distance: FeatureDistance) -> float:
        sound_a = self._clts.sound(ipa_a)
        sound_b = self._clts.sound(ipa_b)
        if sound_a is None or sound_b is None:
            return float(max(len(ipa_a), len(ipa_b)))
        vec_a = np.array(sound_a.feature_matrix())
        vec_b = np.array(sound_b.feature_matrix())
        if feature_distance == FeatureDistance.EUCLIDEAN:
            return float(np.linalg.norm(vec_a - vec_b))
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        cosine = float(np.dot(vec_a, vec_b) / denom)
        return 1 - cosine


class PhoneticDistanceCalculator:
    """Combines segmental and tone distances into a single score."""

    def __init__(self, language_map: Dict[str, str]) -> None:
        self.transcriber = IPATranscriber(language_map)
        self.segmental = SegmentalDistanceCalculator()

    def distance(self, text_a: str, text_b: str, language: str, config: DistanceConfig) -> DistanceBreakdown:
        ipa_a = self.transcriber.transliterate(text_a, language)
        ipa_b = self.transcriber.transliterate(text_b, language)
        stripped_a = strip_tone_marks(ipa_a)
        stripped_b = strip_tone_marks(ipa_b)

        segmental_total, detail = self.segmental.compute(stripped_a, stripped_b, config.segmental_metrics)
        segmental_total *= config.segmental_scale

        tone_component = 0.0
        if config.tone_weight > 0:
            tone_a = extract_tones(text_a)
            tone_b = extract_tones(text_b)
            tone_component = tone_distance(tone_a, tone_b, config.tone_confusion)

        total = segmental_total + config.tone_weight * tone_component
        detail["segmental::total"] = segmental_total
        detail["tone"] = tone_component
        detail["combined"] = total
        return DistanceBreakdown(segmental=segmental_total, tone=tone_component, total=total, details=detail)
