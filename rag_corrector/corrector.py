"""Entity matching and correction logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .entities import Entity
from .phonetics import PhoneticEncoder, PhoneticEncoding


@dataclass(frozen=True)
class Correction:
    """Represents a replacement applied to a transcript."""

    start: int
    end: int
    original: str
    replacement: str
    score: float
    entity: Entity


class EntityCorrector:
    """Match known entities against ASR transcripts using phonetic similarity."""

    def __init__(
        self,
        entities: Sequence[Entity],
        *,
        tone_weight: float = 0.5,
        char_weight: float = 0.4,
        window_slack: int = 1,
        score_threshold: float = 0.45,
    ) -> None:
        if tone_weight < 0:
            raise ValueError("tone_weight must be non-negative")
        if char_weight < 0:
            raise ValueError("char_weight must be non-negative")
        self._entities = list(entities)
        self._tone_weight = tone_weight
        self._char_weight = char_weight
        self._window_slack = max(0, window_slack)
        self._score_threshold = score_threshold
        self._encoder = PhoneticEncoder()
        self._targets: list[tuple[Entity, str, PhoneticEncoding]] = []
        for entity in self._entities:
            self._targets.append((entity, entity.surface, self._encoder.encode(entity.surface)))
            for alias in entity.aliases:
                self._targets.append((entity, alias, self._encoder.encode(alias)))

    @property
    def entities(self) -> Sequence[Entity]:
        return tuple(self._entities)

    def find_corrections(self, transcript: str) -> List[Correction]:
        char_encodings = [self._encoder.encode(char) for char in transcript]
        candidates: list[Correction] = []

        for entity, text, target in self._targets:
            target_len = len(text)
            min_len = max(1, target_len - self._window_slack)
            max_len = target_len + self._window_slack

            for window_length in range(min_len, max_len + 1):
                for start in range(0, len(transcript) - window_length + 1):
                    end = start + window_length
                    original = transcript[start:end]
                    if original == entity.surface:
                        continue
                    segment_encoding = self._encoder.combine(char_encodings[start:end])
                    phonetic_score = self._encoder.distance(segment_encoding, target, self._tone_weight)
                    char_score = _normalized_edit_distance(original, text)
                    score = phonetic_score + self._char_weight * char_score
                    if score <= self._score_threshold:
                        candidates.append(
                            Correction(
                                start=start,
                                end=end,
                                original=original,
                                replacement=entity.surface,
                                score=score,
                                entity=entity,
                            )
                        )

        candidates.sort(key=lambda c: (c.score, c.start, -(c.end - c.start)))
        selected: list[Correction] = []
        occupied: list[tuple[int, int]] = []
        for candidate in candidates:
            if any(not (candidate.end <= s or candidate.start >= e) for s, e in occupied):
                continue
            occupied.append((candidate.start, candidate.end))
            selected.append(candidate)
        return selected

    def apply(self, transcript: str, corrections: Sequence[Correction]) -> str:
        if not corrections:
            return transcript
        chars = list(transcript)
        result = []
        last_idx = 0
        for correction in sorted(corrections, key=lambda c: c.start):
            result.append("".join(chars[last_idx:correction.start]))
            result.append(correction.replacement)
            last_idx = correction.end
        result.append("".join(chars[last_idx:]))
        return "".join(result)

    def correct(self, transcript: str) -> tuple[str, List[Correction]]:
        corrections = self.find_corrections(transcript)
        return self.apply(transcript, corrections), corrections


def _normalized_edit_distance(source: str, target: str) -> float:
    if not source and not target:
        return 0.0
    distance = _levenshtein_text(source, target)
    return distance / max(len(source), len(target), 1)


def _levenshtein_text(source: str, target: str) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)
    prev = list(range(len(target) + 1))
    for i, sc in enumerate(source, start=1):
        curr = [i]
        for j, tc in enumerate(target, start=1):
            cost = 0 if sc == tc else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]
