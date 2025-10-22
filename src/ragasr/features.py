"""Feature extraction for IPA sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from panphon import FeatureTable

_ft = FeatureTable()
_VECTOR_MAP = {"+": 1.0, "-": -1.0, "0": 0.0}


@dataclass(slots=True)
class PhoneticRepresentation:
    """Holds intermediate phonetic representations."""

    ipa: str
    segments: List[str]
    features: List[np.ndarray]
    tones: List[int]
    stresses: List[int]


def ipa_to_segments(ipa: str) -> List[str]:
    return _ft.ipa_segs(ipa)


def segments_to_features(segments: List[str]) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    dim = len(_ft.names)
    for seg in segments:
        try:
            raw = _ft.segment_to_vector(seg)
        except Exception:
            raw = None
        if isinstance(raw, (list, tuple)) and len(raw) == dim:
            vector = np.array([_VECTOR_MAP.get(val, 0.0) for val in raw], dtype=float)
        else:
            vector = np.zeros(dim, dtype=float)
        feats.append(vector)
    return feats


__all__ = [
    "PhoneticRepresentation",
    "ipa_to_segments",
    "segments_to_features",
]
