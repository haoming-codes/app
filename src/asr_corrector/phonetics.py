"""Utilities for phonetic representations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from phonemizer.phonemize import phonemize
from pypinyin import Style, lazy_pinyin

LETTER_NAMES = {
    "A": "ay",
    "B": "bee",
    "C": "see",
    "D": "dee",
    "E": "ee",
    "F": "ef",
    "G": "jee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}

IPA_TONE_PATTERN = re.compile(r"[˥˦˧˨˩˥˩]|[¹²³⁴⁵]|[0-5]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class PhoneticRepresentation:
    """Holds IPA and tone information."""

    ipa: str
    detoned_ipa: str
    tones: Tuple[str, ...]


@lru_cache(maxsize=2048)
def _phonemize(text: str, language: str) -> str:
    punctuation = ';:,.!?¡¿—…"\'-()'
    return phonemize(
        text,
        language=language,
        backend='espeak',
        strip=True,
        punctuation_marks=punctuation,
    )


def _expand_acronym(text: str) -> str:
    tokens: List[str] = []
    for ch in text:
        if ch in LETTER_NAMES:
            tokens.append(LETTER_NAMES[ch])
        else:
            tokens.append(ch)
    return " ".join(tokens)


def is_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def normalize_text_for_phonemization(text: str) -> str:
    if text.isupper() and len(text) <= 10:
        return _expand_acronym(text)
    return text


def ipa_for_text(text: str) -> PhoneticRepresentation:
    normalized = normalize_text_for_phonemization(text)
    language = "cmn" if is_cjk(text) else "en-us"
    ipa = _phonemize(normalized, language)
    detoned = IPA_TONE_PATTERN.sub("", ipa)
    tones = tuple(_extract_tones(text)) if is_cjk(text) else tuple()
    return PhoneticRepresentation(ipa=ipa, detoned_ipa=detoned, tones=tones)


def _extract_tones(text: str) -> Iterable[str]:
    tones = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    for syllable in tones:
        digit = syllable[-1]
        yield digit if digit.isdigit() else "5"


def tone_distance(
    tones_a: Sequence[str],
    tones_b: Sequence[str],
    default_cost: float,
    confusion_costs: dict | None = None,
) -> float:
    length = max(len(tones_a), len(tones_b))
    if length == 0:
        return 0.0
    total = 0.0
    for a, b in zip(_pad(tones_a, length), _pad(tones_b, length)):
        if a == b:
            continue
        cost = default_cost
        if confusion_costs:
            cost = confusion_costs.get(a, {}).get(b, cost)
        total += cost
    return total / length


def _pad(seq: Sequence[str], length: int) -> Sequence[str]:
    padded = list(seq)
    while len(padded) < length:
        padded.append("5")
    return padded


class CLTSSpace:
    """Provides CLTS feature vectors for IPA segments."""

    def __init__(self) -> None:
        from pyclts import CLTS

        self.clts = CLTS()
        self._zero_vec = np.array(self.clts.bipa["a"].numeric, dtype=float) * 0

    @lru_cache(maxsize=2048)
    def vectorize(self, ipa: str) -> np.ndarray:
        features: List[float] = []
        for sound in self.clts.bipa.segment_string(ipa):
            try:
                entry = self.clts.bipa[sound]
                features.extend(float(v) for v in entry.numeric)
            except KeyError:
                features.extend(self._zero_vec.tolist())
        if not features:
            return np.zeros_like(self._zero_vec)
        return np.array(features, dtype=float)

    def distance(self, ipa_a: str, ipa_b: str) -> float:
        vec_a = self.vectorize(ipa_a)
        vec_b = self.vectorize(ipa_b)
        length = max(vec_a.size, vec_b.size)
        if vec_a.size < length:
            vec_a = np.pad(vec_a, (0, length - vec_a.size))
        if vec_b.size < length:
            vec_b = np.pad(vec_b, (0, length - vec_b.size))
        return float(np.linalg.norm(vec_a - vec_b))
