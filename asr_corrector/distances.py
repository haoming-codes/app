"""Distance metrics for phonetic and tonal similarity."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from nltk.metrics.aline import Aline
from panphon.distance import Distance as PanphonDistance

from .phonetics import PhoneticSequence


@lru_cache(maxsize=1)
def _aline() -> Aline:
    return Aline()


@lru_cache(maxsize=1)
def _panphon() -> PanphonDistance:
    return PanphonDistance()


def segmental_distance(
    seq1: PhoneticSequence,
    seq2: PhoneticSequence,
    metric: str = "panphon",
) -> float:
    """Compute a normalized segmental distance between two IPA sequences."""

    ipa1 = seq1.as_ipa()
    ipa2 = seq2.as_ipa()
    if metric == "aline":
        score = _aline().distance(ipa1, ipa2)
    elif metric == "panphon":
        score = _panphon().distance(ipa1, ipa2)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported distance metric: {metric}")

    normaliser = max(len(seq1.segments), len(seq2.segments), 1)
    return score / normaliser


def tone_distance(
    tones1: Sequence[int],
    tones2: Sequence[int],
    mismatch_penalty: float = 1.0,
) -> float:
    """Compute a normalized tone distance between two tone sequences."""

    shared = min(len(tones1), len(tones2))
    total = 0.0
    for t1, t2 in zip(tones1[:shared], tones2[:shared]):
        total += abs(t1 - t2) / 4.0
    total += abs(len(tones1) - len(tones2)) * mismatch_penalty
    normaliser = max(len(tones1), len(tones2), 1)
    return total / normaliser


def combined_distance(
    seq1: PhoneticSequence,
    seq2: PhoneticSequence,
    metric: str = "panphon",
    segmental_weight: float = 1.0,
    lambda_tone: float = 0.5,
    tone_mismatch_penalty: float = 1.0,
) -> float:
    """Combine segmental and tonal distances into a single score."""

    seg = segmental_distance(seq1, seq2, metric=metric)
    tone = tone_distance(seq1.tones, seq2.tones, mismatch_penalty=tone_mismatch_penalty)
    return segmental_weight * seg + lambda_tone * tone
