"""ASR correction utilities based on phonetic distance."""
from __future__ import annotations

from typing import Optional

from .config import (
    CorrectionConfig,
    DistanceConfig,
    KnowledgeBase,
    KnowledgeBaseEntry,
    SegmentalMetricConfig,
    ToneDistanceConfig,
    default_distance_config,
)
from .corrector import ASRCorrector, CorrectionSuggestion
from .distances import DistanceCombiner
from .phonetics import PhoneticTranscriber

__all__ = [
    "ASRCorrector",
    "CorrectionSuggestion",
    "CorrectionConfig",
    "DistanceConfig",
    "SegmentalMetricConfig",
    "ToneDistanceConfig",
    "default_distance_config",
    "KnowledgeBase",
    "KnowledgeBaseEntry",
    "compute_distance",
]


def compute_distance(
    text_a: str,
    text_b: str,
    *,
    distance_config: Optional[DistanceConfig] = None,
    language_a: Optional[str] = None,
    language_b: Optional[str] = None,
    is_acronym_a: bool = False,
    is_acronym_b: bool = False,
    transcriber: Optional[PhoneticTranscriber] = None,
) -> float:
    """Compute a combined phonetic distance between two substrings.

    Parameters
    ----------
    text_a, text_b:
        Input substrings to compare.
    distance_config:
        Optional :class:`DistanceConfig` describing which metrics to use.
    language_a, language_b:
        Optional language hints forwarded to the phonemizer.
    is_acronym_a, is_acronym_b:
        Whether to treat the substrings as spelled-out acronyms.
    transcriber:
        Optional shared :class:`PhoneticTranscriber` instance.
    """

    transcriber = transcriber or PhoneticTranscriber()
    config = distance_config or default_distance_config()
    combiner = DistanceCombiner(config)
    transcription_a = transcriber.transcribe(text_a, language=language_a, is_acronym=is_acronym_a)
    transcription_b = transcriber.transcribe(text_b, language=language_b, is_acronym=is_acronym_b)
    return combiner.distance(
        transcription_a.ipa,
        transcription_b.ipa,
        transcription_a.tone_sequence,
        transcription_b.tone_sequence,
    )
