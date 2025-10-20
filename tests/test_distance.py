import math

import pytest

from asr_corrector import (
    DistanceCalculator,
    DistanceConfig,
    SegmentalConfig,
    SegmentalMetric,
    ToneConfig,
    ToneMetric,
)
from asr_corrector.config import DEFAULT_TONE_CONFUSION


@pytest.mark.parametrize(
    "metric",
    [
        SegmentalMetric.PANPHON,
        SegmentalMetric.ABYDOS_PHONETIC,
        SegmentalMetric.ABYDOS_ALINE,
    ],
)
def test_distance_metrics(metric):
    config = DistanceConfig(
        segmental=SegmentalConfig(metric=metric),
        tone=ToneConfig(metric=ToneMetric.CONFUSION, confusion_costs=DEFAULT_TONE_CONFUSION),
        tone_weight=0.2,
    )
    calculator = DistanceCalculator(config)
    distance = calculator.distance(
        source="阿里巴巴",
        target="阿里爸爸",
        source_language="cmn-Hans",
    )
    assert distance >= 0
    assert calculator.explain(
        source="quantum annealing",
        target="quantum anilling",
        source_language="eng-Latn",
    )["combined"] >= 0


def test_acronym_distance():
    config = DistanceConfig(segmental=SegmentalConfig(metric=SegmentalMetric.ABYDOS_PHONETIC))
    calculator = DistanceCalculator(config)
    distance = calculator.distance(
        source="AG AL",
        target="AG.AL",
        source_language="eng-Latn",
        treat_target_as_acronym=True,
    )
    other = calculator.distance(
        source="agal",
        target="AG.AL",
        source_language="eng-Latn",
        treat_target_as_acronym=True,
    )
    assert distance < other


def test_tone_distance_weighting():
    config = DistanceConfig(
        tone=ToneConfig(metric=ToneMetric.CONFUSION, confusion_costs=DEFAULT_TONE_CONFUSION),
        tone_weight=1.0,
    )
    calculator = DistanceCalculator(config)
    same = calculator.distance("妈妈", "妈妈", source_language="cmn-Hans")
    diff = calculator.distance("妈妈", "麻麻", source_language="cmn-Hans")
    assert same <= diff


def test_clts_metric():
    pytest.importorskip("clts")
    config = DistanceConfig(
        segmental=SegmentalConfig(metric=SegmentalMetric.CLTS_VECTOR, clts_vector_distance="cosine"),
        tone=ToneConfig(metric=ToneMetric.NONE),
    )
    calculator = DistanceCalculator(config)
    distance = calculator.distance(
        source="quantum", target="quantam", source_language="eng-Latn"
    )
    assert distance >= 0
