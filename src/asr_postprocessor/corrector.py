"""Core correction logic for phonetic named-entity repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .entities import NamedEntity
from .phonetics import CharacterPhonetic, ChinesePhoneticizer, PhoneticDistance


@dataclass
class LexiconEntry:
    entity: NamedEntity
    ipa: str
    ipa_token_count: int


class PhoneticLexicon:
    """Container for the user-supplied named-entity list."""

    def __init__(self, phoneticizer: ChinesePhoneticizer) -> None:
        self._phoneticizer = phoneticizer
        self._entries: List[LexiconEntry] = []

    def add(self, entity: NamedEntity) -> None:
        ipa = entity.ipa if entity.ipa is not None else self._phoneticizer.transliterate(entity.surface)
        ipa = ipa.strip()
        ipa_token_count = max(1, len(ipa.split())) if ipa else len(entity.surface)
        self._entries.append(LexiconEntry(entity=entity, ipa=ipa, ipa_token_count=ipa_token_count))

    def extend(self, entities: Iterable[NamedEntity]) -> None:
        for entity in entities:
            self.add(entity)

    @property
    def entries(self) -> Sequence[LexiconEntry]:
        return tuple(self._entries)


@dataclass
class CorrectionConfig:
    """Configuration hyper-parameters controlling the correction behaviour."""

    threshold: float = 0.75
    length_slack: int = 1
    tradeoff_lambda: float = 0.85
    length_penalty: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= self.tradeoff_lambda <= 1.0:
            raise ValueError("tradeoff_lambda must be in [0, 1]")
        if self.length_slack < 0:
            raise ValueError("length_slack must be >= 0")
        if self.length_penalty < 0:
            raise ValueError("length_penalty must be >= 0")


@dataclass
class CorrectionCandidate:
    entity: NamedEntity
    start_index: int
    end_index: int
    matched_text: str
    score: float


class CorrectionEngine:
    """Perform phonetic correction on ASR transcriptions."""

    def __init__(
        self,
        lexicon: PhoneticLexicon,
        phoneticizer: Optional[ChinesePhoneticizer] = None,
        distance: Optional[PhoneticDistance] = None,
        config: Optional[CorrectionConfig] = None,
    ) -> None:
        self._phoneticizer = phoneticizer or ChinesePhoneticizer()
        self._distance = distance or PhoneticDistance()
        self._lexicon = lexicon
        self._config = config or CorrectionConfig()

    @property
    def config(self) -> CorrectionConfig:
        return self._config

    def generate_candidates(self, text: str) -> List[CorrectionCandidate]:
        tokens = self._phoneticizer.ipa_tokens(text)
        if not tokens:
            return []

        candidates: List[CorrectionCandidate] = []
        for entry in self._lexicon.entries:
            lengths = self._candidate_window_lengths(entry.ipa_token_count)
            for window_len in lengths:
                if window_len <= 0 or window_len > len(tokens):
                    continue
                for start in range(0, len(tokens) - window_len + 1):
                    window = tokens[start : start + window_len]
                    candidate = self._score_candidate(text, entry, window)
                    if candidate is not None and candidate.score >= self._config.threshold:
                        candidates.append(candidate)
        return candidates

    def correct(self, text: str) -> Tuple[str, List[CorrectionCandidate]]:
        candidates = self.generate_candidates(text)
        if not candidates:
            return text, []

        selected = self._select_non_overlapping(candidates)
        if not selected:
            return text, []

        corrected = self._apply_replacements(text, selected)
        return corrected, selected

    def _candidate_window_lengths(self, target_len: int) -> Sequence[int]:
        slack = self._config.length_slack
        return range(max(1, target_len - slack), target_len + slack + 1)

    def _score_candidate(
        self,
        text: str,
        entry: LexiconEntry,
        window: Sequence[CharacterPhonetic],
    ) -> Optional[CorrectionCandidate]:
        ipa_window = " ".join(token.ipa for token in window)
        phonetic_similarity = self._distance.similarity(entry.ipa, ipa_window)

        target_len = entry.ipa_token_count
        observed_len = len(window)
        length_delta = abs(target_len - observed_len)
        length_score = pow(2.718281828459045, -self._config.length_penalty * length_delta)

        lambda_ = self._config.tradeoff_lambda
        score = lambda_ * phonetic_similarity + (1.0 - lambda_) * length_score

        start_index = window[0].index
        end_index = window[-1].index
        matched_text = text[start_index : end_index + 1]

        return CorrectionCandidate(
            entity=entry.entity,
            start_index=start_index,
            end_index=end_index,
            matched_text=matched_text,
            score=score,
        )

    def _select_non_overlapping(self, candidates: Sequence[CorrectionCandidate]) -> List[CorrectionCandidate]:
        sorted_candidates = sorted(candidates, key=lambda c: (-c.score, c.start_index, -(c.end_index - c.start_index)))
        selected: List[CorrectionCandidate] = []
        occupied: List[Tuple[int, int]] = []

        for candidate in sorted_candidates:
            if any(not (candidate.end_index < start or candidate.start_index > end) for start, end in occupied):
                continue
            selected.append(candidate)
            occupied.append((candidate.start_index, candidate.end_index))
        return sorted(selected, key=lambda c: c.start_index)

    def _apply_replacements(self, text: str, candidates: Sequence[CorrectionCandidate]) -> str:
        result_parts: List[str] = []
        last_index = 0
        for candidate in candidates:
            result_parts.append(text[last_index : candidate.start_index])
            result_parts.append(candidate.entity.surface)
            last_index = candidate.end_index + 1
        result_parts.append(text[last_index:])
        return "".join(result_parts)
