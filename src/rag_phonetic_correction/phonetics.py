"""Mandarin phonetic conversion utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from pypinyin import Style, pinyin

__all__ = ["MandarinPhonetics", "SyllablePhonetics"]


_PY_TONE_RE = re.compile(r"(?P<base>[a-züv]+)(?P<tone>[0-5])?$")

IPA_INITIALS = {
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

IPA_FINALS = {
    "a": "a",
    "o": "u̯ɔ",
    "e": "ɤ",
    "ê": "ɛ",
    "ai": "aɪ̯",
    "ei": "eɪ̯",
    "ao": "aʊ̯",
    "ou": "oʊ̯",
    "an": "an",
    "en": "ən",
    "ang": "ɑŋ",
    "eng": "əŋ",
    "ong": "ʊŋ",
    "er": "aɻ",
    "i": "i",
    "ia": "ja",
    "ie": "jɛ",
    "iao": "jɑʊ̯",
    "iu": "joʊ̯",
    "iou": "joʊ̯",
    "ian": "jɛn",
    "in": "in",
    "iang": "jɑŋ",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "io": "jo",
    "ua": "wa",
    "uo": "wo",
    "u": "u",
    "uai": "waɪ̯",
    "ui": "weɪ̯",
    "uei": "weɪ̯",
    "uan": "wan",
    "un": "wən",
    "uen": "wən",
    "uang": "wɑŋ",
    "ueng": "wəŋ",
    "v": "y",
    "ve": "yɛ",
    "van": "ɥɛn",
    "vn": "yn",
    "vng": "yŋ",
    "ue": "ɥe",
}

RETROFLEX_FINALS = {"zh", "ch", "sh", "r"}
DENTAL_FINALS = {"z", "c", "s"}

ZERO_INITIAL_FINAL_MAP = {
    "yi": "i",
    "ya": "ia",
    "yo": "io",
    "ye": "ie",
    "yao": "iao",
    "you": "iou",
    "yan": "ian",
    "yin": "in",
    "yang": "iang",
    "ying": "ing",
    "yong": "iong",
    "yu": "v",
    "yue": "ve",
    "yuan": "van",
    "yun": "vn",
    "wu": "u",
    "wa": "ua",
    "wo": "uo",
    "wai": "uai",
    "wei": "uei",
    "wan": "uan",
    "wen": "uen",
    "wang": "uang",
    "weng": "ueng",
}


@dataclass
class SyllablePhonetics:
    """Represents the phonetic annotation of a single syllable."""

    surface: Optional[str]
    pinyin: str
    ipa: str
    tone: int


class MandarinPhonetics:
    """Convert Mandarin text or Pinyin sequences into IPA representations."""

    def analyze_text(self, text: str) -> List[SyllablePhonetics]:
        initials = pinyin(
            text,
            style=Style.INITIALS,
            strict=False,
            errors="default",
        )
        finals = pinyin(
            text,
            style=Style.FINALS_TONE3,
            strict=False,
            errors="default",
            neutral_tone_with_five=True,
        )
        syllables = pinyin(
            text,
            style=Style.TONE3,
            strict=False,
            errors="default",
            neutral_tone_with_five=True,
        )
        results: List[SyllablePhonetics] = []
        for char, ini_list, fin_list, syl_list in zip(text, initials, finals, syllables):
            initial = ini_list[0] if ini_list else ""
            final = fin_list[0] if fin_list else ""
            syllable = syl_list[0] if syl_list else ""
            info = self._syllable_from_components(
                syllable=syllable,
                initial=initial,
                final=final,
                surface=char,
            )
            if info is None:
                info = SyllablePhonetics(surface=char, pinyin=syllable, ipa="", tone=5)
            results.append(info)
        return results

    def analyze_pinyin(self, pronunciation: Sequence[str]) -> List[SyllablePhonetics]:
        results: List[SyllablePhonetics] = []
        for token in pronunciation:
            token = token.strip()
            if not token:
                continue
            info = self._syllable_from_components(syllable=token)
            if info:
                results.append(info)
        return results

    def _syllable_from_components(
        self,
        syllable: str,
        initial: Optional[str] = None,
        final: Optional[str] = None,
        surface: Optional[str] = None,
    ) -> Optional[SyllablePhonetics]:
        syllable = syllable.strip()
        normalized = syllable.lower()
        match = _PY_TONE_RE.match(normalized)
        if not match:
            return None
        base = match.group("base").replace("ü", "v")
        tone = match.group("tone")
        tone_number = int(tone) if tone else 5
        initial = (initial or "").strip().lower()
        final = (final or "").strip().lower()

        if not final:
            final = base[len(initial) :] if initial else base
            if not final and initial:
                # Syllables like "n2" where the consonant functions as the nucleus.
                final = initial
                initial = ""

        if not initial:
            zero_initial = ZERO_INITIAL_FINAL_MAP.get(base)
            if zero_initial:
                final = zero_initial
            else:
                initial, final = self._split_without_initial(base)

        final_base = final.rstrip("12345")
        ipa_initial = IPA_INITIALS.get(initial, "")
        ipa_final = self._ipa_for_final(final_base, initial)
        if not ipa_final and ipa_initial == "":
            # Nothing useful to work with.
            return None

        ipa = (ipa_initial + ipa_final).strip()
        return SyllablePhonetics(surface=surface, pinyin=syllable, ipa=ipa, tone=tone_number)

    def _ipa_for_final(self, final: str, initial: str) -> str:
        if final == "i" and initial in RETROFLEX_FINALS:
            return "ɨ"
        if final == "i" and initial in DENTAL_FINALS:
            return "ɿ"
        if initial in {"j", "q", "x"} and final.startswith("u"):
            final = "v" + final[1:]
        ipa = IPA_FINALS.get(final)
        if ipa:
            return ipa
        # Fall back to a best-effort conversion: treat remaining vowels as is.
        return final

    def _split_without_initial(self, base: str) -> tuple[str, str]:
        if base in ZERO_INITIAL_FINAL_MAP:
            return "", ZERO_INITIAL_FINAL_MAP[base]
        for ini in sorted(IPA_INITIALS.keys(), key=len, reverse=True):
            if base.startswith(ini):
                final = base[len(ini) :]
                if final:
                    return ini, final
        return "", base


def tones_from_syllables(syllables: Iterable[SyllablePhonetics]) -> List[int]:
    return [s.tone for s in syllables]

