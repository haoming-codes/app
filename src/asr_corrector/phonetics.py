from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from .tones import strip_tone_number, tone_sequence_from_syllables

try:
    from pypinyin import Style, pinyin
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise ImportError("pypinyin is required for phonetic conversion") from exc

try:
    from pyclts.transcriptionsystem import TranscriptionSystem
except ImportError:  # pragma: no cover - optional dependency
    TranscriptionSystem = None  # type: ignore


@dataclass(frozen=True)
class PhoneticRepresentation:
    text: str
    syllables: Tuple[str, ...]
    ipa: Tuple[str, ...]
    ipa_detoned: Tuple[str, ...]
    tones: Tuple[int, ...]


class PhoneticConverter:
    """Convert Chinese text to syllables, IPA representations, and tone sequences."""

    def __init__(
        self,
        neutral_tone_with_five: bool = True,
        use_strict_segmentation: bool = True,
    ) -> None:
        self.neutral_tone_with_five = neutral_tone_with_five
        self.use_strict_segmentation = use_strict_segmentation
        self._clts_source = None
        self._clts_target = None
        if TranscriptionSystem is not None:
            try:
                self._clts_source = TranscriptionSystem("pinyin")
                self._clts_target = TranscriptionSystem("ipa")
            except (KeyError, TypeError):  # pragma: no cover - CLTS not installed
                self._clts_source = None
                self._clts_target = None

    def _pinyin_syllables(self, text: str) -> Tuple[str, ...]:
        style = Style.TONE3 if self.neutral_tone_with_five else Style.TONE
        try:
            result = pinyin(
                text,
                style=style,
                neutral_tone_with_five=self.neutral_tone_with_five,
                strict=self.use_strict_segmentation,
            )
        except TypeError:  # pragma: no cover - legacy pypinyin without strict flag
            result = pinyin(
                text,
                style=style,
                neutral_tone_with_five=self.neutral_tone_with_five,
            )
        flattened: List[str] = [syllable for group in result for syllable in group if syllable]
        return tuple(flattened)

    def _ipa_from_pinyin(self, syllable: str) -> str:
        if self._clts_source is not None and self._clts_target is not None:
            try:
                return str(self._clts_source[syllable].convert(self._clts_target))
            except (KeyError, ValueError):
                pass
        return strip_tone_number(syllable)

    def _detoned_syllable(self, syllable: str) -> str:
        return strip_tone_number(syllable)

    @lru_cache(maxsize=2048)
    def convert(self, text: str) -> PhoneticRepresentation:
        syllables = self._pinyin_syllables(text)
        ipa = tuple(self._ipa_from_pinyin(syl) for syl in syllables)
        ipa_detoned = tuple(self._ipa_from_pinyin(self._detoned_syllable(syl)) for syl in syllables)
        tones = tone_sequence_from_syllables(syllables)
        return PhoneticRepresentation(text=text, syllables=syllables, ipa=ipa, ipa_detoned=ipa_detoned, tones=tones)


def join_segments(segments: Sequence[str]) -> str:
    return " ".join(seg for seg in segments if seg)
