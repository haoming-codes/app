from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

ToneSequence = Tuple[int, ...]


DEFAULT_TONE_CONFUSION: Mapping[Tuple[int, int], float] = {
    (1, 2): 0.6,
    (2, 1): 0.6,
    (2, 3): 0.7,
    (3, 2): 0.7,
    (3, 4): 0.6,
    (4, 3): 0.6,
    (1, 4): 0.9,
    (4, 1): 0.9,
    (1, 3): 0.8,
    (3, 1): 0.8,
    (2, 4): 0.9,
    (4, 2): 0.9,
    (1, 5): 0.5,
    (5, 1): 0.5,
    (2, 5): 0.5,
    (5, 2): 0.5,
    (3, 5): 0.5,
    (5, 3): 0.5,
    (4, 5): 0.5,
    (5, 4): 0.5,
}


@dataclass(frozen=True)
class ToneDistance:
    """Weighted edit distance between Mandarin tone sequences."""

    substitution_costs: Mapping[Tuple[int, int], float] = DEFAULT_TONE_CONFUSION
    deletion_cost: float = 1.0
    insertion_cost: float = 1.0

    def substitution(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        if (a, b) in self.substitution_costs:
            return self.substitution_costs[(a, b)]
        if (b, a) in self.substitution_costs:
            return self.substitution_costs[(b, a)]
        return 1.0 + 0.1 * abs(a - b)

    def distance(self, seq1: Sequence[int], seq2: Sequence[int]) -> float:
        m, n = len(seq1), len(seq2)
        if m == 0:
            return float(n) * self.insertion_cost
        if n == 0:
            return float(m) * self.deletion_cost
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i * self.deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = j * self.insertion_cost
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost_sub = dp[i - 1][j - 1] + self.substitution(seq1[i - 1], seq2[j - 1])
                cost_del = dp[i - 1][j] + self.deletion_cost
                cost_ins = dp[i][j - 1] + self.insertion_cost
                dp[i][j] = min(cost_sub, cost_del, cost_ins)
        return dp[m][n]

    def normalized_distance(self, seq1: Sequence[int], seq2: Sequence[int]) -> float:
        denom = max(len(seq1), len(seq2), 1)
        return self.distance(seq1, seq2) / float(denom)


def extract_tone_number(syllable: str, default: int = 5) -> int:
    for char in reversed(syllable):
        if char.isdigit():
            return int(char)
    return default


def strip_tone_number(syllable: str) -> str:
    return "".join(ch for ch in syllable if not ch.isdigit())


def tone_sequence_from_syllables(syllables: Iterable[str], default: int = 5) -> ToneSequence:
    return tuple(extract_tone_number(s, default=default) for s in syllables)
