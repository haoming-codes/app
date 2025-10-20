"""Utilities for working with Mandarin tones."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pypinyin import Style, pinyin

from .config import ToneConfusionMatrix


@dataclass
class ToneSequence:
    """Represents a sequence of Mandarin tone numbers."""

    tones: List[int]

    def pad_to_length(self, length: int) -> None:
        if len(self.tones) < length:
            self.tones.extend([5] * (length - len(self.tones)))


def extract_tones(text: str) -> ToneSequence:
    """Extract tone numbers from a Mandarin string using pypinyin."""

    tone_numbers: List[int] = []
    raw = pinyin(text, style=Style.TONE3, heteronym=False)
    for syllables in raw:
        for syllable in syllables:
            tone = 5  # neutral tone default
            if syllable:
                digit = syllable[-1]
                if digit.isdigit():
                    tone = int(digit)
            tone_numbers.append(tone)
    return ToneSequence(tone_numbers)


def tone_distance(source: ToneSequence, target: ToneSequence, confusion: ToneConfusionMatrix) -> float:
    """Compute a confusion-weighted tone distance between two sequences."""

    if not source.tones and not target.tones:
        return 0.0

    max_len = max(len(source.tones), len(target.tones))
    s = source.tones + [5] * (max_len - len(source.tones))
    t = target.tones + [5] * (max_len - len(target.tones))

    total = 0.0
    for s_tone, t_tone in zip(s, t):
        total += confusion.cost(s_tone, t_tone)
    return total / max_len
