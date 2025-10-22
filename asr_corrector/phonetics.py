from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from panphon.featuretable import FeatureTable
from phonemizer import phonemize
from phonemizer.separator import Separator


_ACRONYM_IPA = {
    "A": "eɪ",
    "B": "b iː",
    "C": "s iː",
    "D": "d iː",
    "E": "iː",
    "F": "ɛ f",
    "G": "dʒ iː",
    "H": "eɪ tʃ",
    "I": "aɪ",
    "J": "dʒ eɪ",
    "K": "k eɪ",
    "L": "ɛ l",
    "M": "ɛ m",
    "N": "ɛ n",
    "O": "oʊ",
    "P": "p iː",
    "Q": "k j uː",
    "R": "ɑː ɹ",
    "S": "ɛ s",
    "T": "t iː",
    "U": "j uː",
    "V": "v iː",
    "W": "d ʌ b əl j uː",
    "X": "ɛ k s",
    "Y": "w aɪ",
    "Z": "z iː",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[A-Za-z]+")
_TONE_RE = re.compile(r"[1-5]")
_STRESS_MAP = {"ˈ": 2, "ˌ": 1}


def _vectorize(values: List[str]) -> List[int]:
    mapping = {"+": 1, "-": -1, "0": 0}
    return [mapping.get(v, 0) for v in values]


@dataclass
class PhoneticRepresentation:
    ipa: str
    segments: List[str]
    features: List[List[int]]
    tones: List[int]
    stresses: List[int]


class PhoneticTranscriber:
    """Transcribes multilingual text into IPA and articulatory features."""

    def __init__(self) -> None:
        self._separator = Separator(word="", syllable="", phone=" ")
        self._feature_table = FeatureTable()

    @property
    def feature_table(self) -> FeatureTable:
        return self._feature_table

    def transcribe(self, text: str) -> PhoneticRepresentation:
        segments: List[str] = []
        tones: List[int] = []
        stresses: List[int] = []

        for token, kind in self._tokenize(text):
            if kind == "acronym":
                token_segments = self._transcribe_acronym(token)
            elif kind == "zh":
                token_segments, token_tones = self._transcribe_chinese(token)
                tones.extend(token_tones)
            elif kind == "en":
                token_segments, token_stresses = self._transcribe_english(token)
                stresses.extend(token_stresses)
            else:
                token_segments = []

            segments.extend(token_segments)

        clean_segments: List[str] = [seg for seg in segments if seg]
        fine_segments: List[str] = []
        for seg in clean_segments:
            fine_segments.extend(self._feature_table.ipa_segs(seg))
        ipa = "".join(fine_segments)
        features = [
            _vectorize(self._feature_table.segment_to_vector(seg))
            for seg in fine_segments
        ]

        return PhoneticRepresentation(
            ipa=ipa,
            segments=fine_segments,
            features=features,
            tones=tones,
            stresses=stresses,
        )

    def ipa(self, text: str) -> str:
        return self.transcribe(text).ipa

    def _tokenize(self, text: str) -> Iterable[tuple[str, str]]:
        pos = 0
        while pos < len(text):
            char = text[pos]
            if _CJK_RE.match(char):
                end = pos
                while end < len(text) and _CJK_RE.match(text[end]):
                    end += 1
                yield text[pos:end], "zh"
                pos = end
                continue
            if char.isalpha():
                end = pos
                while end < len(text) and text[end].isalpha():
                    end += 1
                token = text[pos:end]
                kind = "acronym" if self._is_acronym(token) else "en"
                yield token, kind
                pos = end
                continue
            pos += 1

    def _is_acronym(self, token: str) -> bool:
        letters = [c for c in token if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters) and len(letters) > 1

    def _transcribe_acronym(self, token: str) -> List[str]:
        segments: List[str] = []
        for char in token:
            if char.isalpha():
                ipa = _ACRONYM_IPA.get(char.upper())
                if ipa:
                    segments.extend(ipa.split())
        return segments

    def _transcribe_chinese(self, token: str) -> tuple[List[str], List[int]]:
        phon = phonemize(
            token,
            language="cmn-latn-pinyin",
            backend="espeak",
            strip=True,
            with_stress=False,
            separator=self._separator,
            language_switch="remove-flags",
        )
        segments: List[str] = []
        tones: List[int] = []
        for part in phon.split():
            digits = _TONE_RE.findall(part)
            if digits:
                tones.append(int(digits[-1]))
                part = _TONE_RE.sub("", part)
            if part:
                segments.append(part)
        return segments, tones

    def _transcribe_english(self, token: str) -> tuple[List[str], List[int]]:
        phon = phonemize(
            token,
            language="en-us",
            backend="espeak",
            strip=True,
            with_stress=True,
            separator=self._separator,
            language_switch="remove-flags",
        )
        segments: List[str] = []
        stresses: List[int] = []
        for part in phon.split():
            for stress_char, value in _STRESS_MAP.items():
                if stress_char in part:
                    stresses.append(value)
                    part = part.replace(stress_char, "")
            if part:
                segments.append(part)
        return segments, stresses


__all__ = ["PhoneticTranscriber", "PhoneticRepresentation"]
