"""Phonetic encoding helpers for Chinese text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency is not available."""


@dataclass(frozen=True)
class PhoneticEncoding:
    """Represents a per-character IPA encoding."""

    text: str
    segments: List[str]

    def window(self, start: int, end: int) -> List[str]:
        """Return the IPA segments covering ``text[start:end]``."""

        return self.segments[start:end]


class PhoneticEncoder:
    """Convert Chinese text to IPA strings using :mod:`pypinyin`."""

    def __init__(self, *, errors: str = "keep") -> None:
        try:
            from pypinyin import Style, pinyin
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise MissingDependencyError(
                "pypinyin is required for phonetic encoding. Install it with 'pip install pypinyin'."
            ) from exc

        def ipa_pinyin(
            text: str,
            *,
            strict: bool,
            errors: Callable[[str], List[str]] | str,
        ) -> List[List[str]]:
            """Proxy to :func:`pypinyin.pinyin` with IPA output."""

            return pinyin(
                text,
                style=Style.IPA,
                heteronym=True,
                strict=strict,
                errors=errors,
            )

        self._ipa_pinyin: Callable[..., List[List[str]]] = ipa_pinyin
        self._errors = errors

    def encode(self, text: str) -> PhoneticEncoding:
        """Return the IPA representation for every character in ``text``."""

        if not isinstance(text, str):  # pragma: no cover - defensive branch
            raise TypeError("text must be a string")

        ipa_matrix = self._ipa_pinyin(
            text,
            strict=False,
            errors=self._error_handler if self._errors == "keep" else self._errors,
        )
        segments: List[str] = []
        for syllables in ipa_matrix:
            if not syllables:
                segments.append("")
                continue
            # ``pinyin`` returns a list of alternatives; we select the first.
            segments.append(syllables[0])
        return PhoneticEncoding(text=text, segments=segments)

    @staticmethod
    def _error_handler(char: str) -> List[str]:
        """Keep non-Hanzi characters untouched so we preserve alignment."""

        return [char]


__all__ = ["PhoneticEncoder", "PhoneticEncoding", "MissingDependencyError"]
