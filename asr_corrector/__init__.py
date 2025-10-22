from __future__ import annotations

from .config import DistanceConfig, DistanceWeights, SegmentDistanceConfig, SegmentMetricConfig
from .correction import CorrectionCandidate, CorrectionEngine, CorrectionResult
from .distance import DistanceBreakdown, DistanceCalculator
from .phonetics import PhoneticRepresentation, PhoneticTranscriber

_default_transcriber: PhoneticTranscriber | None = None


def _get_transcriber(transcriber: PhoneticTranscriber | None = None) -> PhoneticTranscriber:
    global _default_transcriber
    if transcriber is not None:
        return transcriber
    if _default_transcriber is None:
        _default_transcriber = PhoneticTranscriber()
    return _default_transcriber


def transcribe_to_ipa(text: str, *, transcriber: PhoneticTranscriber | None = None) -> str:
    """Return the IPA representation of ``text``."""

    return _get_transcriber(transcriber).ipa(text)


def compute_distance(
    first: str,
    second: str,
    *,
    config: DistanceConfig | None = None,
    transcriber: PhoneticTranscriber | None = None,
) -> DistanceBreakdown:
    """Compute the configured distance between two strings."""

    calculator = DistanceCalculator(config=config, transcriber=_get_transcriber(transcriber))
    return calculator.compute(first, second)


__all__ = [
    "CorrectionCandidate",
    "CorrectionEngine",
    "CorrectionResult",
    "DistanceBreakdown",
    "DistanceCalculator",
    "DistanceConfig",
    "DistanceWeights",
    "PhoneticRepresentation",
    "PhoneticTranscriber",
    "SegmentDistanceConfig",
    "SegmentMetricConfig",
    "compute_distance",
    "transcribe_to_ipa",
]
