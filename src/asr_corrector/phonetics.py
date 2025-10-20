"""Utilities for phonetic transcription and normalisation."""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from phonemizer import phonemize
from phonemizer.separator import Separator
from pypinyin import Style, lazy_pinyin

_LOGGER = logging.getLogger(__name__)

_CHINESE_RANGE = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)


def is_chinese_char(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CHINESE_RANGE)


def _split_unicode_blocks(text: str) -> List[str]:
    tokens: List[str] = []
    buffer: List[str] = []
    buffer_chinese = False

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            tokens.append("".join(buffer).strip())
            buffer = []

    for char in text:
        if char.isspace():
            flush_buffer()
            buffer_chinese = False
            continue
        char_is_chinese = is_chinese_char(char)
        if not buffer:
            buffer.append(char)
            buffer_chinese = char_is_chinese
            continue
        if char_is_chinese != buffer_chinese:
            flush_buffer()
            buffer.append(char)
            buffer_chinese = char_is_chinese
        else:
            buffer.append(char)
    flush_buffer()
    return [token for token in tokens if token]


def normalise_acronym(term: str) -> str:
    letters = re.findall(r"[A-Za-z]", term)
    return " ".join(letters)


@dataclass
class TranscriptionResult:
    text: str
    ipa: str
    tone_sequence: Optional[List[str]]


class PhoneticTranscriber:
    """Handles conversion between surface forms and IPA/tone representations."""

    def __init__(self, language_hint: Optional[str] = None):
        self.language_hint = language_hint
        self.separator = Separator(word=" ", syllable=" ", phone=" ")

    @functools.lru_cache(maxsize=2048)
    def _phonemise(self, text: str, language: str) -> str:
        try:
            return phonemize(
                text,
                language=language,
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                separator=self.separator,
            )
        except Exception as exc:  # pragma: no cover - backend errors are logged
            _LOGGER.warning("Phonemizer failed for '%s' (%s)", text, exc)
            return ""

    def _tone_sequence(self, text: str) -> Optional[List[str]]:
        if not any(is_chinese_char(ch) for ch in text):
            return None
        return lazy_pinyin(text, style=Style.TONE3)

    def transcribe(self, text: str, language: Optional[str] = None, *, is_acronym: bool = False) -> TranscriptionResult:
        if is_acronym:
            text = normalise_acronym(text)
            language = language or "en-us"
        if not text:
            return TranscriptionResult(text="", ipa="", tone_sequence=None)

        language = language or self._detect_language(text)
        ipa = self._phonemise(text, language)
        tone_seq = self._tone_sequence(text)
        return TranscriptionResult(text=text, ipa=ipa, tone_sequence=tone_seq)

    def _detect_language(self, text: str) -> str:
        if any(is_chinese_char(ch) for ch in text):
            return "zh"
        return self.language_hint or "en-us"

    def segment_tokens(self, text: str) -> List[str]:
        blocks = _split_unicode_blocks(text)
        tokens: List[str] = []
        for block in blocks:
            if any(is_chinese_char(ch) for ch in block):
                tokens.extend(list(block))
            else:
                tokens.extend(block.split())
        return [tok for tok in tokens if tok]

    def windows(self, tokens: List[str], min_len: int, max_len: int) -> Iterable[tuple[int, int, str]]:
        length = len(tokens)
        for start in range(length):
            for end in range(start + min_len, min(start + max_len, length) + 1):
                window_tokens = tokens[start:end]
                if window_tokens and all(len(tok) == 1 and is_chinese_char(tok) for tok in window_tokens):
                    text = "".join(window_tokens)
                else:
                    text = " ".join(window_tokens)
                yield start, end, text
