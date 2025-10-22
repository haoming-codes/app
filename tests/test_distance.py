from asr_corrector.config import DistanceConfig
from asr_corrector.distance import DistanceCalculator


def test_identical_strings_zero_distance():
    calc = DistanceCalculator()
    result = calc.compute("Mini Map", "Mini Map")
    assert result.combined < 1e-6
    assert result.segment < 1e-6
    assert result.feature < 1e-6


def test_alternative_configuration():
    config = DistanceConfig(segment_metric="aline", feature_metric="euclidean")
    calc = DistanceCalculator(config=config)
    result = calc.compute("Mini Map", "minivan")
    assert result.combined >= 0.0
    assert result.segment >= 0.0
    assert result.feature >= 0.0
