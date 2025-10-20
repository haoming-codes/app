"""Mandarin phonetic transcription utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from panphon.distance import Distance
from panphon.featuretable import FeatureTable
from pypinyin import Style, pinyin


TONE_MARKS = {
    "1": "˥",
    "2": "˧˥",
    "3": "˨˩˦",
    "4": "˥˩",
    "5": "˧",
}

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
    "r": "ʐ",
    "z": "ts",
    "c": "tsʰ",
    "s": "s",
    "y": "j",
    "w": "w",
}

FINAL_IPA = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ai": "aɪ̯",
    "ei": "eɪ̯",
    "ao": "ɑʊ̯",
    "ou": "oʊ̯",
    "an": "an",
    "en": "ən",
    "ang": "ɑŋ",
    "eng": "ɤŋ",
    "ong": "ʊŋ",
    "er": "ɚ",
    "ia": "ja",
    "iao": "jɑʊ̯",
    "ie": "jɛ",
    "iu": "joʊ̯",
    "ian": "jɛn",
    "iang": "jɑŋ",
    "in": "in",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "io": "jo",
    "i": "i",
    "ua": "wa",
    "uo": "wɔ",
    "uai": "waɪ̯",
    "ui": "weɪ̯",
    "uan": "wan",
    "uang": "wɑŋ",
    "un": "wən",
    "ueng": "uəŋ",
    "u": "u",
    "ue": "we",
    "uen": "wən",
    "van": "ɥɛn",
    "ve": "ɥe",
    "vn": "yn",
    "v": "y",
}

ALVEO_PALATALS = {"j", "q", "x", "y"}
RETROFLEXES = {"zh", "ch", "sh", "r"}
DENTALS = {"z", "c", "s"}


@dataclass
class Transcription:
    ipa_syllables: List[str]

    @property
    def ipa(self) -> str:
        return " ".join(self.ipa_syllables)

    def __iter__(self):
        return iter(self.ipa_syllables)


class MandarinTranscriber:
    """Convert Chinese text to IPA approximations for distance computation."""

    def __init__(self, strict: bool = False) -> None:
        self._distance = Distance()
        self._feature_table = FeatureTable()
        self.strict = strict

    def transcribe(self, text: str) -> Transcription:
        initials = pinyin(text, style=Style.INITIALS, strict=False)
        finals = pinyin(text, style=Style.FINALS_TONE3, strict=False)
        ipa_syllables: List[str] = []
        for ini_list, fin_list in zip(initials, finals):
            initial = ini_list[0]
            final = fin_list[0]
            ipa_syllables.append(self._syllable_to_ipa(initial, final))
        return Transcription(ipa_syllables)

    def normalized_distance(self, a: str, b: str) -> float:
        """Compute normalized phonetic distance between two IPA strings."""
        raw = self._distance.weighted_feature_edit_distance(a, b)
        length = max(len(self._feature_table.segs_safe(a)), len(self._feature_table.segs_safe(b)))
        if length == 0:
            return 0.0
        return raw / length

    def _syllable_to_ipa(self, initial: str, final: str) -> str:
        tone = "5"
        if final and final[-1].isdigit():
            tone = final[-1]
            base_final = final[:-1]
        else:
            base_final = final
        base_final = base_final or ""
        base_final = base_final.replace("v", "ü")

        ipa_initial = INITIAL_IPA.get(initial, initial)
        ipa_final = self._final_to_ipa(base_final, initial)
        tone_mark = TONE_MARKS.get(tone, TONE_MARKS["5"])
        parts = [part for part in (ipa_initial, ipa_final, tone_mark) if part]
        return " ".join(parts)

    def _final_to_ipa(self, final: str, initial: str) -> str:
        if not final:
            return ""
        # Special handling for apical vowels and alveolo-palatal interactions.
        if final == "i":
            if initial in RETROFLEXES:
                return "ɻ̩"
            if initial in DENTALS:
                return "ɿ"
            return "i"
        if final in {"u", "uan", "un", "ue"} and initial in ALVEO_PALATALS:
            replacements = {
                "u": "y",
                "uan": "ɥɛn",
                "un": "yn",
                "ue": "ɥe",
            }
            return replacements[final]
        if final == "ong" and initial in ALVEO_PALATALS:
            return "ʊŋ"
        if final == "ü":
            return "y"
        if final == "üe":
            return "ɥe"
        if final == "üan":
            return "ɥɛn"
        if final == "ün":
            return "yn"

        ipa = FINAL_IPA.get(final)
        if ipa is not None:
            return ipa
        if not self.strict:
            return final
        raise KeyError(f"No IPA mapping for final '{final}'")


def syllable_distance(a: Sequence[str], b: Sequence[str], transcriber: MandarinTranscriber) -> float:
    """Convenience helper to compute normalized distance between syllable lists."""
    return transcriber.normalized_distance(" ".join(a), " ".join(b))
