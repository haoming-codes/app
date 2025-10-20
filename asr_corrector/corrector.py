"""Correction pipeline for ASR outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .entities import EntityLexicon
from .matcher import DistanceMetric
from .phonetics import BasePhoneticEncoder


@dataclass(frozen=True)
class Correction:
    """Represents a single replacement in the ASR output."""

    start: int
    end: int
    original: str
    replacement: str
    distance: float
    similarity: float

    def overlaps(self, other: "Correction") -> bool:
        return not (self.end <= other.start or self.start >= other.end)


class CorrectionResult:
    """Stores the corrections found for an utterance."""

    def __init__(self, original_text: str, corrections: Iterable[Correction]):
        self.original_text = original_text
        self.corrections: List[Correction] = sorted(corrections, key=lambda c: c.start)

    def apply(self) -> str:
        """Return the corrected text after applying all replacements."""

        if not self.corrections:
            return self.original_text
        pieces: List[str] = []
        cursor = 0
        for correction in self.corrections:
            pieces.append(self.original_text[cursor:correction.start])
            pieces.append(correction.replacement)
            cursor = correction.end
        pieces.append(self.original_text[cursor:])
        return "".join(pieces)

    @property
    def corrected_text(self) -> str:
        return self.apply()


class NameCorrectionPipeline:
    """Pipeline that replaces entity-like substrings using phonetic similarity."""

    def __init__(
        self,
        lexicon: EntityLexicon,
        encoder: BasePhoneticEncoder,
        distance_metric: DistanceMetric,
        distance_threshold: float = 2.5,
        min_similarity: float = 0.0,
    ):
        self._lexicon = lexicon
        self._encoder = encoder
        self._distance_metric = distance_metric
        self._distance_threshold = distance_threshold
        self._min_similarity = min_similarity

    def _generate_candidates(self, text: str) -> List[Correction]:
        candidates: List[Correction] = []
        max_len = self._lexicon.max_surface_length
        ipa_cache: dict[str, str] = {}
        for start in range(len(text)):
            for length in range(1, max_len + 1):
                end = start + length
                fragment = text[start:end]
                if len(fragment) < length:
                    break
                if not fragment.strip():
                    continue
                ipa = ipa_cache.get(fragment)
                if ipa is None:
                    ipa = self._encoder.encode(fragment)
                    ipa_cache[fragment] = ipa
                best_entity = None
                best_distance = float("inf")
                for entity in self._lexicon.entities:
                    distance = self._distance_metric.distance(ipa, entity.ipa)
                    if distance < best_distance:
                        best_distance = distance
                        best_entity = entity
                if best_entity is None:
                    continue
                if best_entity.surface == fragment:
                    continue
                if best_distance > self._distance_threshold:
                    continue
                similarity = 1.0 / (1.0 + best_distance)
                if similarity < self._min_similarity:
                    continue
                candidates.append(
                    Correction(
                        start=start,
                        end=end,
                        original=fragment,
                        replacement=best_entity.surface,
                        distance=best_distance,
                        similarity=similarity,
                    )
                )
        return candidates

    def correct(self, text: str) -> CorrectionResult:
        candidates = self._generate_candidates(text)
        selected: List[Correction] = []
        occupied_positions: set[int] = set()
        for candidate in sorted(candidates, key=lambda c: (c.similarity, -(c.end - c.start)), reverse=True):
            if any(pos in occupied_positions for pos in range(candidate.start, candidate.end)):
                continue
            selected.append(candidate)
            occupied_positions.update(range(candidate.start, candidate.end))
        selected.sort(key=lambda c: c.start)
        return CorrectionResult(text, selected)


__all__ = ["Correction", "CorrectionResult", "NameCorrectionPipeline"]
