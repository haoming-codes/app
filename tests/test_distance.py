import pytest

from phonetic_rag.config import DistanceAggregationConfig
from phonetic_rag.distance import PhoneticDistanceCalculator


def test_identical_strings_have_zero_distance():
    calc = PhoneticDistanceCalculator(DistanceAggregationConfig())
    breakdown = calc.distance('美国', '美国')
    assert breakdown.total == pytest.approx(0.0, abs=1e-12)


def test_misrecognition_distance_smaller_for_similar_words():
    calc = PhoneticDistanceCalculator(DistanceAggregationConfig())
    close = calc.distance('Mini Map', 'minivan')
    far = calc.distance('Mini Map', '美国')
    assert close.total < 0.5
    assert far.total > close.total
