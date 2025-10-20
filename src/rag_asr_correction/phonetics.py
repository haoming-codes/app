"""Phonetic utilities for ASR correction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

from phonemizer import phonemize
from phonemizer.separator import Separator
from pypinyin import Style, pinyin


@dataclass(frozen=True)
class PhoneticRepresentation:
    """Stores phonetic representations used for distance calculation."""

    text: str
    language: str
    ipa: str
    segmental: str
    tone_sequence: Tuple[int, ...]


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_ACRONYM_PATTERN = re.compile(r"^[A-Z0-9\.]+$")
_TONE_MARKS = {
    "\u02E5",
    "\u02E6",
    "\u02E7",
    "\u02E8",
    "\u02E9",
    "\u0304",
    "\u0300",
    "\u0301",
    "\u030C",
    "\u0302",
    "\u0342",
    "\u035B",
}


def detect_language(text: str) -> str:
    """Detect whether the text is Mandarin Chinese or English."""

    if _CJK_PATTERN.search(text):
        return "cmn"
    return "en-us"


def _normalize_acronym(text: str) -> str:
    """Expand acronym strings into space separated letters."""

    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    if not cleaned:
        return text
    expanded: List[str] = []
    for char in cleaned:
        expanded.append(char)
    return " ".join(expanded)


def _normalize_text(text: str, language: str) -> str:
    """Normalize text before phonemization."""

    if language.startswith("cmn"):
        return text
    if text.isupper() and _ACRONYM_PATTERN.match(text):
        return _normalize_acronym(text)
    return text


def strip_tone_marks(ipa: str) -> str:
    """Remove tone marks from an IPA string."""

    normalized = unicodedata.normalize("NFD", ipa)
    filtered = [
        ch
        for ch in normalized
        if not (
            unicodedata.category(ch).startswith("M") and ch in _TONE_MARKS
        )
    ]
    return unicodedata.normalize("NFC", "".join(filtered))


def tokenize_for_language(text: str, language: str) -> List[str]:
    """Tokenize text into a coarse sequence for window matching."""

    if language.startswith("cmn"):
        return [ch for ch in text if not ch.isspace()]
    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)
    return tokens or [text]


def _tones_from_text(text: str, language: str) -> Tuple[int, ...]:
    """Extract tone sequence from text."""

    if not language.startswith("cmn"):
        return tuple()

    result: List[int] = []
    syllables = pinyin(text, style=Style.TONE3, errors="ignore", strict=False)
    for syllable_group in syllables:
        for syllable in syllable_group:
            match = re.search(r"([1-5])", syllable)
            if match:
                result.append(int(match.group(1)))
            else:
                result.append(5)
    return tuple(result)


def build_phonetic_representation(text: str, language: str | None = None) -> PhoneticRepresentation:
    """Construct the phonetic representation for ``text``."""

    if language is None:
        language = detect_language(text)
    normalized = _normalize_text(text, language)
    ipa = phonemize(
        normalized,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        separator=Separator(phone="", syllable="", word=" "),
        njobs=1,
    )
    segmental = strip_tone_marks(ipa.replace(" ", ""))
    tones = _tones_from_text(text, language)
    return PhoneticRepresentation(
        text=text,
        language=language,
        ipa=ipa,
        segmental=segmental,
        tone_sequence=tones,
    )


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    is_space: bool


_TOKEN_PATTERN = re.compile(
    r"\s+|[\u4e00-\u9fff]+|[A-Za-z0-9']+|[^\w\s]",
)


def tokenize_with_spans(text: str) -> List[Token]:
    """Tokenize text while preserving spans."""

    tokens: List[Token] = []
    for match in _TOKEN_PATTERN.finditer(text):
        segment = match.group(0)
        if segment.isspace():
            tokens.append(
                Token(
                    text=segment,
                    start=match.start(),
                    end=match.end(),
                    is_space=True,
                )
            )
            continue

        if all(_CJK_PATTERN.match(ch) for ch in segment):
            for offset, ch in enumerate(segment):
                tokens.append(
                    Token(
                        text=ch,
                        start=match.start() + offset,
                        end=match.start() + offset + 1,
                        is_space=False,
                    )
                )
            continue

        tokens.append(
            Token(
                text=segment,
                start=match.start(),
                end=match.end(),
                is_space=False,
            )
        )
    return tokens
