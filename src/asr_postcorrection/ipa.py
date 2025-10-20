"""Utilities for converting Chinese text to pseudo-IPA strings for similarity scoring."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

from pypinyin import Style, lazy_pinyin


_PINYIN_RE = re.compile(r"[1-5]")


def _normalize_pinyin(syllable: str) -> str:
    base = _PINYIN_RE.sub("", syllable.lower())
    base = base.replace("ü", "v")
    base = base.replace("u:", "v")
    return base


def _pinyin_to_pseudo_ipa(syllable: str) -> str:
    s = _normalize_pinyin(syllable)
    # Finals adjustments (longest first)
    replacements = [
        ("iang", "jɑŋ"),
        ("iong", "jʊŋ"),
        ("uang", "wɑŋ"),
        ("ueng", "wəŋ"),
        ("ang", "ɑŋ"),
        ("eng", "əŋ"),
        ("ing", "iŋ"),
        ("ong", "ʊŋ"),
        ("iao", "jau"),
        ("ian", "jɛn"),
        ("ien", "jɛn"),
        ("ua", "wa"),
        ("uo", "wo"),
        ("uai", "wai"),
        ("ui", "wei"),
        ("iu", "jou"),
        ("ie", "jɛ"),
        ("ia", "ja"),
        ("ve", "yɛ"),
        ("van", "yɛn"),
        ("vn", "yn"),
        ("er", "aɻ"),
        ("ong", "ʊŋ"),
    ]
    for target, repl in replacements:
        if target in s:
            s = s.replace(target, repl)
    # Initial adjustments
    initial_replacements = [
        ("zh", "ʈʂ"),
        ("ch", "ʈʂʰ"),
        ("sh", "ʂ"),
    ]
    for target, repl in initial_replacements:
        if s.startswith(target):
            s = repl + s[len(target) :]
            break
    s = s.replace("q", "tɕʰ")
    s = s.replace("x", "ɕ")
    s = s.replace("j", "tɕ")
    if s.startswith("r"):
        s = s.replace("r", "ʐ", 1)
    s = s.replace("z", "ts")
    s = s.replace("c", "tsʰ")
    s = s.replace("y", "j")
    s = s.replace("w", "w")
    s = s.replace("ng", "ŋ")
    return s


def _characters(text: str) -> List[str]:
    return [char for char in text]


@lru_cache(maxsize=8192)
def char_to_ipa(char: str) -> str:
    """Converts a single Chinese character to its pseudo-IPA representation."""
    if not char.strip():
        return char
    syllables = lazy_pinyin(
        char,
        style=Style.TONE3,
        strict=False,
        errors="default",
    )
    if not syllables:
        return char
    return "".join(_pinyin_to_pseudo_ipa(s) for s in syllables)


def text_to_ipa_sequence(text: str) -> List[str]:
    """Returns a per-character IPA sequence for the provided text."""
    return [char_to_ipa(ch) for ch in _characters(text)]


def text_to_ipa(text: str) -> str:
    """Converts arbitrary Chinese text to a concatenated IPA string."""
    return "".join(text_to_ipa_sequence(text))


def ipa_sequence_to_string(ipa_sequence: Iterable[str]) -> str:
    """Utility to concatenate an IPA sequence."""
    return "".join(ipa_sequence)
