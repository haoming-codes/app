"""Distance helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len_a][len_b]


@dataclass(slots=True)
class NormalizedEditDistance:
    """Compute a normalized edit distance between integer sequences."""

    def __call__(self, seq_a: Sequence[int], seq_b: Sequence[int]) -> float:
        if not seq_a and not seq_b:
            return 0.0
        distance = _edit_distance(seq_a, seq_b)
        denom = max(len(seq_a), len(seq_b))
        return distance / denom
