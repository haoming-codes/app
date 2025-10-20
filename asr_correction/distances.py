from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance
from panphon.featuretable import FeatureTable

from .config import DistanceComponentConfig
from .phonetics import PhoneticRepresentation, text_to_representation


@dataclass
class DistanceComponent:
    name: str
    weight: float
    compute: Callable[[PhoneticRepresentation, PhoneticRepresentation], float]


_PANPHON_DISTANCE = PanphonDistance()
_FEATURE_TABLE = FeatureTable()


def _panphon_distance(rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> float:
    return float(_PANPHON_DISTANCE.weighted_feature_edit_distance(rep_a.detoned, rep_b.detoned))


def _phonetic_edit_distance(rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation, **kwargs) -> float:
    ped = PhoneticEditDistance(**kwargs)
    return float(ped.dist(rep_a.detoned, rep_b.detoned))


def _aline_distance(rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation, **kwargs) -> float:
    aline = ALINE(**kwargs)
    return float(aline.dist(rep_a.detoned, rep_b.detoned))


def _clts_distance(rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation) -> float:
    vec_a = _FEATURE_TABLE.word_to_vector_list(rep_a.detoned)
    vec_b = _FEATURE_TABLE.word_to_vector_list(rep_b.detoned)
    if not vec_a or not vec_b:
        return float("inf")
    mean_a = np.mean(np.array(vec_a), axis=0)
    mean_b = np.mean(np.array(vec_b), axis=0)
    return float(np.linalg.norm(mean_a - mean_b))


def _tone_distance(rep_a: PhoneticRepresentation, rep_b: PhoneticRepresentation, matrix: np.ndarray | None = None) -> float:
    tones_a = rep_a.tones
    tones_b = rep_b.tones
    if not tones_a and not tones_b:
        return 0.0
    if len(tones_a) != len(tones_b):
        max_len = max(len(tones_a), len(tones_b))
        tones_a = tones_a + (5,) * (max_len - len(tones_a))
        tones_b = tones_b + (5,) * (max_len - len(tones_b))
    if matrix is None:
        matrix = np.array(
            [
                [0.0, 1.5, 2.0, 2.0, 2.5],
                [1.5, 0.0, 1.5, 2.0, 2.0],
                [2.0, 1.5, 0.0, 1.5, 2.0],
                [2.0, 2.0, 1.5, 0.0, 1.5],
                [2.5, 2.0, 2.0, 1.5, 0.0],
            ]
        )
    matrix = np.asarray(matrix)
    total = 0.0
    for a, b in zip(tones_a, tones_b):
        idx_a = min(max(a, 1), 5) - 1
        idx_b = min(max(b, 1), 5) - 1
        total += matrix[idx_a, idx_b]
    return total / len(tones_a)


class PhoneticDistanceCalculator:
    """Aggregate multiple phonetic distance functions."""

    def __init__(self, components: Sequence[DistanceComponent], tone_tradeoff: float = 1.0) -> None:
        self.components = list(components)
        self.tone_tradeoff = tone_tradeoff

    def representation(self, text: str, language: str) -> PhoneticRepresentation:
        return text_to_representation(text, language)

    def compute(self, text_a: str, text_b: str, language: str) -> float:
        rep_a = self.representation(text_a, language)
        rep_b = self.representation(text_b, language)
        total_weight = 0.0
        total_distance = 0.0
        for component in self.components:
            weight = component.weight
            if component.name == "tone":
                weight *= self.tone_tradeoff
            distance = component.compute(rep_a, rep_b)
            total_distance += weight * distance
            total_weight += weight
        if total_weight == 0.0:
            return total_distance
        return total_distance / total_weight


def build_components(configs: Sequence[DistanceComponentConfig]) -> List[DistanceComponent]:
    components: List[DistanceComponent] = []
    for cfg in configs:
        name = cfg.name
        weight = cfg.weight
        options = cfg.options
        if name == "panphon":
            components.append(DistanceComponent(name, weight, _panphon_distance))
        elif name == "phonetic_edit":
            components.append(
                DistanceComponent(name, weight, lambda a, b, opts=options: _phonetic_edit_distance(a, b, **opts))
            )
        elif name == "aline":
            components.append(DistanceComponent(name, weight, lambda a, b, opts=options: _aline_distance(a, b, **opts)))
        elif name == "clts":
            components.append(DistanceComponent(name, weight, _clts_distance))
        elif name == "tone":
            matrix = options.get("matrix") if isinstance(options, dict) else None
            components.append(
                DistanceComponent(
                    name,
                    weight,
                    lambda a, b, mat=matrix: _tone_distance(a, b, matrix=mat),
                )
            )
        else:
            raise ValueError(f"Unknown distance component '{name}'.")
    return components


def compute_distance(
    text_a: str,
    text_b: str,
    language: str,
    configs: Sequence[DistanceComponentConfig],
    tone_tradeoff: float = 1.0,
) -> float:
    calculator = PhoneticDistanceCalculator(build_components(configs), tone_tradeoff=tone_tradeoff)
    return calculator.compute(text_a, text_b, language)
