from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List, Tuple

import regex

try:
    from clts import CLTS
except ImportError:  # pragma: no cover - optional dependency may be unavailable in tests
    CLTS = None  # type: ignore

import epitran
import panphon
from pypinyin import Style, lazy_pinyin

LETTER_TO_IPA = {
    "A": "eɪ",
    "B": "biː",
    "C": "siː",
    "D": "diː",
    "E": "iː",
    "F": "ɛf",
    "G": "dʒiː",
    "H": "eɪtʃ",
    "I": "aɪ",
    "J": "dʒeɪ",
    "K": "keɪ",
    "L": "ɛl",
    "M": "ɛm",
    "N": "ɛn",
    "O": "oʊ",
    "P": "piː",
    "Q": "kjuː",
    "R": "ɑːr",
    "S": "ɛs",
    "T": "tiː",
    "U": "juː",
    "V": "viː",
    "W": "ˈdʌbəljuː",
    "X": "ɛks",
    "Y": "waɪ",
    "Z": "ziː",
}


@dataclass
class IPAResult:
    ipa: str
    tone_sequence: Tuple[str, ...]


@functools.lru_cache(maxsize=32)
def _get_epitran(lang_code: str) -> epitran.Epitran | None:
    try:
        return epitran.Epitran(lang_code)
    except Exception:  # pragma: no cover - fallback without external data
        return None


def _strip_ipa_tones(ipa: str) -> str:
    tone_marks = re.compile(r"[0-9˥˦˧˨˩ˀˁ˔˕˞˟ˠˡˢˣˤːˑˈˌːˑːˑː]", re.UNICODE)
    combining = re.compile(r"[\u0300-\u036f]")
    without_tones = tone_marks.sub("", ipa)
    return combining.sub("", without_tones)


def _mandarin_tones(text: str) -> Tuple[str, ...]:
    tones = []
    for syllable in lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True):
        match = re.search(r"([1-5])", syllable)
        tones.append(match.group(1) if match else "5")
    return tuple(tones)


def _fallback_ipa(text: str, language: str) -> str:
    if language.startswith("cmn"):
        return " ".join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
    if language.startswith("eng"):
        return " ".join(text.lower())
    return text


def _to_ipa_letters(text: str) -> str:
    ipa_segments: List[str] = []
    for char in text:
        if char.upper() in LETTER_TO_IPA:
            ipa_segments.append(LETTER_TO_IPA[char.upper()])
        else:
            ipa_segments.append(char)
    return " ".join(ipa_segments)


def is_acronym(text: str) -> bool:
    stripped = re.sub(r"[^A-Z]", "", text)
    return bool(stripped) and stripped == re.sub(r"[^A-Z]", "", text.upper())


def ipa_and_tones(text: str, language: str, treat_as_acronym: bool = False) -> IPAResult:
    if treat_as_acronym or is_acronym(text):
        ipa = _to_ipa_letters(text)
        tones: Tuple[str, ...] = tuple()
    else:
        epi = _get_epitran(language)
        if epi is not None:
            try:
                ipa = epi.transliterate(text)
            except Exception:  # pragma: no cover - fallback when g2p resources missing
                ipa = _fallback_ipa(text, language)
        else:
            ipa = _fallback_ipa(text, language)
        if language.startswith("cmn"):
            tones = _mandarin_tones(text)
        else:
            tones = tuple()
    return IPAResult(ipa=ipa, tone_sequence=tones)


def detone_ipa(ipa: str) -> str:
    return _strip_ipa_tones(ipa)


_token_pattern = regex.compile(r"\p{IsHan}+|[A-Za-z]+|\d+|[^\s]", regex.UNICODE)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _token_pattern.finditer(text)]


def ipa_to_panphon_vectors(ipa: str, feature_weights: List[float] | None = None) -> List[List[int]]:
    ft = panphon.featuretable.FeatureTable()
    segments = []
    for seg in ipa.split():
        vec = ft.segment_to_vector_list(seg)
        if vec:
            weighted_vec = list(vec[0])
            if feature_weights is not None:
                weighted_vec = [int(val * weight) for val, weight in zip(weighted_vec, feature_weights)]
            segments.append(weighted_vec)
    return segments


class CLTSAccessor:
    def __init__(self) -> None:
        if CLTS is None:  # pragma: no cover - optional dependency may be unavailable
            raise RuntimeError("pyclts is not installed")
        self._clts = CLTS()
        self._bipa = self._clts.bipa

    def ipa_to_vectors(self, ipa: str) -> List[List[float]]:
        tokens = self._bipa.translate(ipa)
        vectors: List[List[float]] = []
        for token in tokens:
            symbol = self._bipa[token]
            if symbol.type == "unknown":
                continue
            if hasattr(symbol, "numeric") and symbol.numeric is not None:
                vectors.append(list(symbol.numeric))
        return vectors
