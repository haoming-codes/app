"""Configuration structures for the ASR correction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


SegmentMetric = Literal["phonetic_edit", "aline"]
FeatureDistance = Literal["l1", "l2", "cosine"]


@dataclass(slots=True)
class DistanceConfig:
    """Configuration controlling distance computations.

    Parameters
    ----------
    segment_metric:
        Name of the phonetic string metric to use. ``"phonetic_edit"`` maps to
        :class:`abydos.distance.PhoneticEditDistance` and ``"aline"`` maps to
        :class:`abydos.distance.ALINE`.
    feature_distance:
        Local distance function for the dynamic time warping step over
        articulatory feature vectors. ``"l1"`` is the default since PanPhon
        encodes features as -1/0/+1 values.
    lambda_segment / lambda_features / lambda_tone / lambda_stress:
        Trade-off weights for each distance component. They should sum to 1 in
        most scenarios but the :class:`DistanceCalculator` does not enforce it.
    tone_penalty:
        Penalty applied when aligned Mandarin tones differ.
    stress_penalty:
        Penalty applied when aligned English stresses differ. Primary and
        secondary stress markers are treated as distinct values.
    threshold:
        Utility threshold that higher-level components such as
        :class:`~asr_correction.matcher.KnowledgeBaseMatcher` can use to decide
        whether two strings are close enough to count as a correction
        candidate.
    max_window_size:
        Default maximum number of tokens that the matcher should consider when
        extracting windows from the ASR hypothesis.
    min_window_size:
        Default minimum number of tokens to form a window. Single-token windows
        are allowed by default.
    """

    segment_metric: SegmentMetric = "phonetic_edit"
    feature_distance: FeatureDistance = "l1"
    lambda_segment: float = 0.4
    lambda_features: float = 0.4
    lambda_tone: float = 0.1
    lambda_stress: float = 0.1
    tone_penalty: float = 1.0
    stress_penalty: float = 1.0
    threshold: float = 0.35
    max_window_size: int = 6
    min_window_size: int = 1
    language_switch_penalty: Optional[float] = None
    """Optional penalty applied when one string contains Mandarin tones and the
    other does not (or vice versa). When ``None`` the calculator simply counts
    the mismatched tones."""


__all__ = ["DistanceConfig", "SegmentMetric", "FeatureDistance"]
