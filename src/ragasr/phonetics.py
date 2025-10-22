"""Multilingual phonetic utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

from phonemizer.backend import EspeakBackend
from pypinyin import lazy_pinyin, Style

_CMN_BACKEND = EspeakBackend("cmn")
_EN_BACKEND = EspeakBackend("en-us")

_ACRONYM_RE = re.compile(r"^(?:[A-Z](?:\.|$))+\Z")
_CJK_RE = re.compile(r"[\u3400-\u9FFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(slots=True)
class Token:
    """Token representation for multilingual text."""

    text: str
    kind: str

    def is_content(self) -> bool:
        return self.kind in {"cjk", "latin", "acronym", "number"}


def _is_cjk(char: str) -> bool:
    return bool(_CJK_RE.fullmatch(char))


def _is_acronym(word: str) -> bool:
    return bool(_ACRONYM_RE.match(word))


def tokenize_text(text: str) -> List[Token]:
    """Tokenize multilingual text into content and separator tokens."""

    tokens: List[Token] = []
    buffer = []
    buffer_kind = None

    def flush_buffer() -> None:
        nonlocal buffer, buffer_kind
        if buffer:
            tokens.append(Token("".join(buffer), buffer_kind or "latin"))
            buffer = []
            buffer_kind = None

    for char in text:
        if char.isspace():
            flush_buffer()
            tokens.append(Token(char, "space"))
            continue
        if _is_cjk(char):
            flush_buffer()
            tokens.append(Token(char, "cjk"))
            continue
        if char.isdigit():
            if buffer_kind == "number":
                buffer.append(char)
            else:
                flush_buffer()
                buffer = [char]
                buffer_kind = "number"
            continue
        if _LATIN_RE.fullmatch(char):
            if buffer_kind in {"latin", "acronym"}:
                buffer.append(char)
            else:
                flush_buffer()
                buffer = [char]
                buffer_kind = "latin"
            continue
        flush_buffer()
        tokens.append(Token(char, "punct"))

    flush_buffer()

    for token in tokens:
        if token.kind == "latin" and _is_acronym(token.text):
            token.kind = "acronym"
    return tokens


def _letter_spelling(word: str) -> str:
    letters = [c for c in word if c.isalpha()]
    return " ".join(letters)


@lru_cache(maxsize=1024)
def _phonemize_en(text: str) -> str:
    return _EN_BACKEND.phonemize([text], strip=True)[0]


@lru_cache(maxsize=1024)
def _phonemize_cmn(text: str) -> str:
    return _CMN_BACKEND.phonemize([text], strip=True)[0]


def ipa_transcription(text: str) -> str:
    """Return the concatenated IPA transcription of multilingual text."""

    tokens = tokenize_text(text)
    ipa_pieces: List[str] = []
    for token in tokens:
        if token.kind == "space":
            continue
        if token.kind == "cjk":
            ipa_pieces.append(_clean_ipa(_phonemize_cmn(token.text)))
        elif token.kind == "latin":
            ipa_pieces.append(_clean_ipa(_phonemize_en(token.text.lower())))
        elif token.kind == "acronym":
            spelled = _letter_spelling(token.text)
            ipa_pieces.append(_clean_ipa(_phonemize_en(spelled.lower())))
        elif token.kind == "number":
            ipa_pieces.append(_clean_ipa(_phonemize_en(token.text)))
        else:
            continue
    return "".join(ipa_pieces)


def tones_for(text: str) -> List[int]:
    """Return Mandarin tone numbers for CJK characters in the text."""

    tones: List[int] = []
    for char in text:
        if _is_cjk(char):
            pinyins = lazy_pinyin(char, style=Style.TONE3, strict=False)
            if pinyins:
                tone = pinyins[0][-1]
                tones.append(int(tone) if tone.isdigit() else 5)
    return tones


def stress_for_ipa(ipa: str) -> List[int]:
    """Extract stress markers from an English IPA string."""

    stresses: List[int] = []
    for symbol in ipa:
        if symbol == "ˈ":
            stresses.append(1)
        elif symbol == "ˌ":
            stresses.append(0)
    return stresses


def _clean_ipa(ipa: str) -> str:
    return re.sub(r"\([^)]*\)", "", ipa).replace(" ", "")


__all__ = [
    "Token",
    "ipa_transcription",
    "tokenize_text",
    "tones_for",
    "stress_for_ipa",
]
