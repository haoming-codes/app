from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from abydos.distance import ALINE, PhoneticEditDistance
from panphon.distance import Distance as PanphonDistance

from .config import DistanceConfig, SegmentalMetric
from .phonetics import CLTSAccessor, detone_ipa, ipa_and_tones


class DistanceCalculator:
    def __init__(self, config: DistanceConfig) -> None:
        self.config = config
        self._panphon_distance = PanphonDistance()
        self._aline_distance = ALINE()
        self._phonetic_edit = PhoneticEditDistance()
        self._clts_accessor: Optional[CLTSAccessor] = None

    def distance(
        self,
        source: str,
        target: str,
        source_language: str,
        target_language: Optional[str] = None,
        treat_target_as_acronym: bool = False,
    ) -> float:
        target_lang = target_language or source_language
        source_repr = ipa_and_tones(source, source_language)
        target_repr = ipa_and_tones(target, target_lang, treat_as_acronym=treat_target_as_acronym)
        segmental = self._segmental_distance(source_repr.ipa, target_repr.ipa)
        tone = self._tone_distance(source_repr.tone_sequence, target_repr.tone_sequence)
        return self.config.segmental_weight * segmental + self.config.tone_weight * tone

    def explain(
        self,
        source: str,
        target: str,
        source_language: str,
        target_language: Optional[str] = None,
        treat_target_as_acronym: bool = False,
    ) -> Dict[str, float]:
        target_lang = target_language or source_language
        source_repr = ipa_and_tones(source, source_language)
        target_repr = ipa_and_tones(target, target_lang, treat_as_acronym=treat_target_as_acronym)
        segmental = self._segmental_distance(source_repr.ipa, target_repr.ipa)
        tone = self._tone_distance(source_repr.tone_sequence, target_repr.tone_sequence)
        return {
            "segmental": segmental,
            "tone": tone,
            "combined": self.config.segmental_weight * segmental + self.config.tone_weight * tone,
            "source_ipa": source_repr.ipa,
            "target_ipa": target_repr.ipa,
        }

    def _segmental_distance(self, source_ipa: str, target_ipa: str) -> float:
        detoned_source = detone_ipa(source_ipa)
        detoned_target = detone_ipa(target_ipa)
        metric = self.config.segmental.metric
        if metric == SegmentalMetric.PANPHON:
            return self._panphon_distance.weighted_feature_edit_distance(
                detoned_source, detoned_target
            )
        if metric == SegmentalMetric.ABYDOS_PHONETIC:
            return self._phonetic_edit.dist(detoned_source, detoned_target)
        if metric == SegmentalMetric.ABYDOS_ALINE:
            return self._aline_distance.dist(detoned_source, detoned_target)
        if metric == SegmentalMetric.CLTS_VECTOR:
            accessor = self._get_clts_accessor()
            source_vec = accessor.ipa_to_vectors(detoned_source)
            target_vec = accessor.ipa_to_vectors(detoned_target)
            return self._vector_distance(source_vec, target_vec, self.config.segmental.clts_vector_distance)
        raise ValueError(f"Unknown metric: {metric}")

    def _get_clts_accessor(self) -> CLTSAccessor:
        if self._clts_accessor is None:
            self._clts_accessor = CLTSAccessor()
        return self._clts_accessor

    def _vector_distance(
        self,
        source_vec: List[List[float]],
        target_vec: List[List[float]],
        metric: str,
    ) -> float:
        if not source_vec or not target_vec:
            return float(max(len(source_vec), len(target_vec)))
        min_len = min(len(source_vec), len(target_vec))
        pad_len = max(len(source_vec), len(target_vec)) - min_len
        source_array = np.array(source_vec[:min_len])
        target_array = np.array(target_vec[:min_len])
        if metric == "cosine":
            cosine = 0.0
            for s, t in zip(source_array, target_array):
                norm_s = np.linalg.norm(s)
                norm_t = np.linalg.norm(t)
                if norm_s == 0 or norm_t == 0:
                    cosine += 1.0
                else:
                    cosine += 1 - float(np.dot(s, t) / (norm_s * norm_t))
            cosine += pad_len
            return float(cosine)
        if metric == "euclidean":
            diff = source_array - target_array
            distance = float(np.linalg.norm(diff, axis=1).sum())
            distance += pad_len
            return float(distance)
        raise ValueError(f"Unsupported vector distance metric: {metric}")

    def _tone_distance(self, source_tones: Tuple[str, ...], target_tones: Tuple[str, ...]) -> float:
        if not source_tones or not target_tones:
            return float(abs(len(source_tones) - len(target_tones)))
        total = 0.0
        max_len = max(len(source_tones), len(target_tones))
        for i in range(max_len):
            s = source_tones[i] if i < len(source_tones) else "5"
            t = target_tones[i] if i < len(target_tones) else "5"
            total += self.config.tone.cost(s, t)
        return float(total)
