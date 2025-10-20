"""Distance calculators for phonetic and tonal representations."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance

from .config import DistanceConfig, SegmentalMetricConfig, ToneDistanceConfig

_LOGGER = logging.getLogger(__name__)

_TONE_PATTERN = re.compile(r"([1-5])$")


def _strip_tone_markers(ipa: str) -> str:
    return re.sub(r"[˥˦˧˨˩˥˩0-9]+", "", ipa)


class _PanphonMetric:
    def __init__(self, params: Optional[Dict[str, object]] = None):
        self.distance = PanphonDistance(**(params or {}))

    def __call__(self, ipa_a: str, ipa_b: str) -> float:
        return self.distance.weighted_feature_edit_distance(ipa_a, ipa_b)


class _AbydosPhoneticEdit:
    def __init__(self, params: Optional[Dict[str, object]] = None):
        self.distance = PhoneticEditDistance(**(params or {}))

    def __call__(self, ipa_a: str, ipa_b: str) -> float:
        return self.distance.dist(ipa_a, ipa_b)


class _AbydosALINE:
    def __init__(self, params: Optional[Dict[str, object]] = None):
        self.distance = ALINE(**(params or {}))

    def __call__(self, ipa_a: str, ipa_b: str) -> float:
        return self.distance.dist(ipa_a, ipa_b)


class _CLTSDistance:
    def __init__(self, params: Optional[Dict[str, object]] = None):
        try:
            from clts import CLTS
        except Exception as exc:  # pragma: no cover - optional dependency
            _LOGGER.warning("CLTS not available: %s", exc)
            self._system = None
            self._bipa = None
            return

        clts_path = (params or {}).get("clts_path")
        try:
            self._system = CLTS(clts_path) if clts_path else CLTS()
            self._bipa = self._system.transcription_system("bipa")
        except Exception as exc:  # pragma: no cover - dataset issues
            _LOGGER.warning("Failed to initialise CLTS: %s", exc)
            self._system = None
            self._bipa = None

    def _vectorise(self, ipa: str) -> np.ndarray:
        if not self._bipa:
            return np.zeros((0,))
        vectors: List[np.ndarray] = []
        for token in ipa.split():
            try:
                segment = self._bipa[token]
            except KeyError:
                continue
            features = segment.featuredict()
            if not features:
                continue
            vector = np.array([float(features.get(key, 0.0)) for key in sorted(features)])
            vectors.append(vector)
        if not vectors:
            return np.zeros((0,))
        return np.mean(vectors, axis=0)

    def __call__(self, ipa_a: str, ipa_b: str) -> float:
        vec_a = self._vectorise(ipa_a)
        vec_b = self._vectorise(ipa_b)
        if vec_a.size == 0 or vec_b.size == 0:
            return 1.0
        return float(np.linalg.norm(vec_a - vec_b, ord=2))


_METRIC_FACTORIES = {
    "panphon_wfed": _PanphonMetric,
    "abydos_phonetic_edit": _AbydosPhoneticEdit,
    "abydos_aline": _AbydosALINE,
    "clts": _CLTSDistance,
}


def _metric_from_config(config: SegmentalMetricConfig):
    try:
        factory = _METRIC_FACTORIES[config.name]
    except KeyError as exc:
        raise ValueError(f"Unknown metric '{config.name}'") from exc
    return factory(config.parameters)


@dataclass
class SegmentalDistanceCalculator:
    configs: Iterable[SegmentalMetricConfig]

    def __post_init__(self):
        self.metrics = [(cfg.weight, _metric_from_config(cfg)) for cfg in self.configs]

    def __call__(self, ipa_a: str, ipa_b: str) -> float:
        toneless_a = _strip_tone_markers(ipa_a)
        toneless_b = _strip_tone_markers(ipa_b)
        weighted: List[float] = []
        total_weight = 0.0
        for weight, metric in self.metrics:
            if weight <= 0:
                continue
            distance = metric(toneless_a, toneless_b)
            weighted.append(weight * distance)
            total_weight += weight
        if not weighted or total_weight == 0:
            return 1.0
        return sum(weighted) / total_weight


class ToneDistanceCalculator:
    def __init__(self, config: ToneDistanceConfig):
        self.config = config

    def __call__(self, tones_a: Optional[List[str]], tones_b: Optional[List[str]]) -> float:
        if not tones_a or not tones_b:
            return 0.0
        length = max(len(tones_a), len(tones_b))
        penalties: List[float] = []
        for idx in range(length):
            tone_a = self._extract_tone(tones_a, idx)
            tone_b = self._extract_tone(tones_b, idx)
            penalties.append(self._penalty(tone_a, tone_b))
        if not penalties:
            return 0.0
        average = sum(penalties) / len(penalties)
        return self.config.weight * average

    def _extract_tone(self, tones: List[str], index: int) -> str:
        if index >= len(tones):
            return "0"
        tone = tones[index]
        match = _TONE_PATTERN.search(tone)
        return match.group(1) if match else "0"

    def _penalty(self, tone_a: str, tone_b: str) -> float:
        if tone_a == tone_b:
            return 0.0
        penalties = self.config.confusion_penalty.get(tone_a)
        if penalties and tone_b in penalties:
            return penalties[tone_b]
        return self.config.default_penalty


class DistanceCombiner:
    def __init__(self, config: DistanceConfig):
        self.segmental = SegmentalDistanceCalculator(config.segmental_metrics)
        self.tone = ToneDistanceCalculator(config.tone_config) if config.tone_config else None
        self.tone_tradeoff = min(max(config.tone_tradeoff, 0.0), 1.0)

    def distance(self, ipa_a: str, ipa_b: str, tones_a: Optional[List[str]], tones_b: Optional[List[str]]) -> float:
        segmental = self.segmental(ipa_a, ipa_b)
        if not self.tone:
            return segmental
        tone_distance = self.tone(tones_a, tones_b)
        return (1 - self.tone_tradeoff) * segmental + self.tone_tradeoff * tone_distance
