"""Tools for correcting Chinese ASR transcripts using phonetic similarity."""

from .entities import EntitySpec, EntityMatch
from .corrector import PhoneticEntityCorrector, CorrectionResult

__all__ = [
    "EntitySpec",
    "EntityMatch",
    "PhoneticEntityCorrector",
    "CorrectionResult",
]
