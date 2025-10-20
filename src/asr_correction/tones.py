"""Tone distance computation."""

from __future__ import annotations

from typing import Sequence

from .config import ToneDistanceConfig


def tone_edit_distance(
    tones_a: Sequence[str], tones_b: Sequence[str], config: ToneDistanceConfig
) -> float:
    """Compute a normalized tone distance using weighted edit distance."""

    len_a = len(tones_a)
    len_b = len(tones_b)
    if len_a == 0 and len_b == 0:
        return 0.0

    dp = [[0.0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i - 1][0] + config.deletion_penalty
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j - 1] + config.insertion_penalty

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            deletion = dp[i - 1][j] + config.deletion_penalty
            insertion = dp[i][j - 1] + config.insertion_penalty
            substitution = dp[i - 1][j - 1] + config.penalty(tones_a[i - 1], tones_b[j - 1])
            dp[i][j] = min(deletion, insertion, substitution)

    normalizer = max(len_a, len_b, 1)
    return dp[len_a][len_b] / normalizer
