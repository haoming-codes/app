"""Utility helpers for multilingual phonetic processing."""

from __future__ import annotations

import functools
import re
from typing import List

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from pypinyin import Style, lazy_pinyin

_ACRONYM_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
_CHINESE_RANGE = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)


def _is_chinese_char(char: str) -> bool:
    code = ord(char)
    for start, end in _CHINESE_RANGE:
        if start <= code <= end:
            return True
    return False


def normalize_acronym(text: str) -> str:
    """Return a whitespace-separated version if the string is an acronym."""

    candidate = text.strip()
    if not candidate:
        return candidate
    if _ACRONYM_PATTERN.match(candidate) and any(ch.isalpha() for ch in candidate):
        letters = [ch for ch in candidate if ch.isalnum()]
        return " ".join(letters)
    return candidate


@functools.lru_cache(maxsize=8)
def _get_backend(language: str) -> EspeakBackend:
    return EspeakBackend(language=language, preserve_punctuation=True)


def _phonemize_chunk(chunk: str, language: str) -> str:
    backend = _get_backend(language)
    separator = Separator(phone=" ", syllable="", word="")
    return backend.phonemize([chunk], strip=True, separator=separator, njobs=1)[0]


def phonemize_text(text: str) -> str:
    """Return a space-delimited IPA representation of the input string."""

    text = normalize_acronym(text)
    pieces: List[str] = []
    buffer: List[str] = []
    buffer_lang: str | None = None

    def flush_buffer() -> None:
        nonlocal buffer
        nonlocal buffer_lang
        if not buffer:
            return
        chunk = "".join(buffer)
        if buffer_lang is None:
            lang = "en-us"
        else:
            lang = buffer_lang
        pieces.append(_phonemize_chunk(chunk, lang))
        buffer = []
        buffer_lang = None

    for char in text:
        if char.isspace():
            flush_buffer()
            continue
        lang = "cmn" if _is_chinese_char(char) else "en-us"
        if buffer_lang is None:
            buffer_lang = lang
            buffer.append(char)
        elif lang == buffer_lang:
            buffer.append(char)
        else:
            flush_buffer()
            buffer_lang = lang
            buffer.append(char)
    flush_buffer()
    result = " ".join(piece for piece in pieces if piece)
    return re.sub(r"\s+", " ", result).strip()


def ipa_to_segments(ipa: str) -> List[str]:
    """Split the IPA string into individual segment symbols."""

    if not ipa:
        return []
    cleaned = re.sub(r"\s+", " ", ipa.strip())
    return cleaned.split(" ")


def mandarin_tone_sequence(text: str) -> List[str]:
    """Return the Mandarin tone sequence using numbered tone notation."""

    normalized = normalize_acronym(text)
    tones: List[str] = []
    syllables = lazy_pinyin(normalized, style=Style.TONE3, strict=False)
    for syllable in syllables:
        match = re.search(r"([1-5])", syllable)
        if match:
            tones.append(match.group(1))
    return tones


def detone_ipa(ipa: str) -> str:
    """Remove tone marks and digits from an IPA transcription."""

    if not ipa:
        return ipa
    return re.sub(r"[˥˦˧˨˩0-9]", "", ipa)


def tokenize_with_spans(text: str) -> List[tuple[str, int, int]]:
    """Tokenize text into Chinese characters or whitespace-delimited chunks."""

    tokens: List[tuple[str, int, int]] = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char.isspace():
            i += 1
            continue
        start = i
        if _is_chinese_char(char):
            tokens.append((char, start, start + 1))
            i += 1
            continue
        while i < length and not text[i].isspace() and not _is_chinese_char(text[i]):
            i += 1
        tokens.append((text[start:i], start, i))
    return tokens
