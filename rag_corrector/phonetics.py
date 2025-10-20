"""Phonetic utilities built on top of :mod:`pypinyin` and :mod:`panphon`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from panphon.distance import Distance
from pypinyin import Style, lazy_pinyin


INITIAL_IPA = {
    "": "",
    "b": "p",
    "p": "pʰ",
    "m": "m",
    "f": "f",
    "d": "t",
    "t": "tʰ",
    "n": "n",
    "l": "l",
    "g": "k",
    "k": "kʰ",
    "h": "x",
    "j": "tɕ",
    "q": "tɕʰ",
    "x": "ɕ",
    "zh": "ʈʂ",
    "ch": "ʈʂʰ",
    "sh": "ʂ",
    "r": "ɻ",
    "z": "ts",
    "c": "tsʰ",
    "s": "s",
}


FINAL_IPA = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ai": "aɪ",
    "ei": "eɪ",
    "ao": "ɑʊ",
    "ou": "oʊ",
    "an": "an",
    "en": "ən",
    "ang": "ɑŋ",
    "eng": "əŋ",
    "ong": "ʊŋ",
    "i": "i",
    "ia": "ja",
    "iao": "jɑʊ",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "ie": "jɛ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "iu": "joʊ",
    "io": "jo",
    "ua": "wa",
    "uo": "wo",
    "uai": "waɪ",
    "ui": "weɪ",
    "uan": "wan",
    "uang": "wɑŋ",
    "un": "wən",
    "ueng": "wəŋ",
    "u": "u",
    "er": "aɻ",
    "v": "y",
    "ve": "yɛ",
    "van": "ɥɛn",
    "vn": "yn",
    "": "",
}


SPECIAL_SYLLABLES = {
    "zhi": "ʈʂɻ̩",
    "chi": "ʈʂʰɻ̩",
    "shi": "ʂɻ̩",
    "ri": "ɻ̩",
    "zi": "tsz̩",
    "ci": "tsʰz̩",
    "si": "sz̩",
}


@dataclass(frozen=True)
class PhoneticEncoding:
    ipa: str
    syllable_ipa: Tuple[str, ...]
    tones: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.syllable_ipa)


class PhoneticEncoder:
    """Convert Chinese text to IPA strings and tone sequences."""

    def __init__(self) -> None:
        self._distance = Distance()

    @staticmethod
    def _split_tone(final_with_tone: str) -> tuple[str, int]:
        tone = 5
        base = final_with_tone
        while base and base[-1].isdigit():
            tone = int(base[-1])
            base = base[:-1]
        return base, tone

    def _syllable_to_ipa(self, initial: str, final_with_tone: str, full_pinyin: str) -> tuple[str, int]:
        base_final, tone = self._split_tone(final_with_tone)
        normalized = (initial or "") + base_final
        if normalized in SPECIAL_SYLLABLES:
            return SPECIAL_SYLLABLES[normalized], tone

        ipa_initial = INITIAL_IPA.get(initial, "")

        ipa_final = None
        if base_final.startswith("u") and initial in {"j", "q", "x"}:
            ipa_final = FINAL_IPA.get("v" + base_final[1:]) or FINAL_IPA.get(base_final[1:])

        if ipa_final is None:
            ipa_final = FINAL_IPA.get(base_final)

        if ipa_final is None:
            ipa_final = base_final or full_pinyin

        return ipa_initial + ipa_final, tone

    def encode_syllables(self, text: str) -> List[tuple[str, str, str]]:
        initials = lazy_pinyin(text, style=Style.INITIALS, errors="default", strict=False)
        finals = lazy_pinyin(text, style=Style.FINALS_TONE3, errors="default", strict=False)
        full = lazy_pinyin(text, style=Style.TONE3, errors="default", strict=False)
        triples: list[tuple[str, str, str]] = []
        for init, fin, syll in zip(initials, finals, full):
            triples.append((init, fin, syll))
        return triples

    @lru_cache(maxsize=2048)
    def encode(self, text: str) -> PhoneticEncoding:
        syllable_ipas: list[str] = []
        tones: list[int] = []
        for initial, final, syllable in self.encode_syllables(text):
            if not final or not initial:
                if syllable and all(ch.isdigit() for ch in syllable):
                    # pypinyin sometimes returns tone only; ignore.
                    continue
            ipa, tone = self._syllable_to_ipa(initial, final, syllable)
            if not ipa:
                ipa = syllable
            syllable_ipas.append(ipa)
            tones.append(tone)
        return PhoneticEncoding(ipa="".join(syllable_ipas), syllable_ipa=tuple(syllable_ipas), tones=tuple(tones))

    def combine(self, encodings: Sequence[PhoneticEncoding]) -> PhoneticEncoding:
        ipa_parts: list[str] = []
        syllable_parts: list[str] = []
        tone_parts: list[int] = []
        for enc in encodings:
            ipa_parts.append(enc.ipa)
            syllable_parts.extend(enc.syllable_ipa)
            tone_parts.extend(enc.tones)
        return PhoneticEncoding("".join(ipa_parts), tuple(syllable_parts), tuple(tone_parts))

    def distance(self, left: PhoneticEncoding, right: PhoneticEncoding, tone_weight: float = 0.5) -> float:
        ipa_distance = self._distance.feature_edit_distance_div_maxlen(left.ipa, right.ipa)
        tone_distance = self._normalized_levenshtein(left.tones, right.tones)
        return ipa_distance + tone_weight * tone_distance

    @staticmethod
    def _normalized_levenshtein(source: Sequence[int], target: Sequence[int]) -> float:
        if not source and not target:
            return 0.0
        distance = _levenshtein(source, target)
        return distance / max(len(source), len(target), 1)


def _levenshtein(source: Sequence[int], target: Sequence[int]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)
    prev = list(range(len(target) + 1))
    for i, s in enumerate(source, start=1):
        curr = [i]
        for j, t in enumerate(target, start=1):
            cost = 0 if s == t else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]
