from asr_correction.config import DistanceConfig
from asr_correction.distance import DistanceCalculator


def test_distance_zero_for_identical_strings():
    calc = DistanceCalculator()
    result = calc.distance('Mini Map', 'Mini Map')
    assert result.overall == 0.0
    assert result.segment == 0.0
    assert result.features == 0.0


def test_distance_with_alternative_metrics():
    config = DistanceConfig(
        segment_metric='aline',
        feature_distance='cosine',
        lambda_segment=0.3,
        lambda_features=0.4,
        lambda_tone=0.2,
        lambda_stress=0.1,
    )
    calc = DistanceCalculator(config=config)
    breakdown = calc.distance('mango', '美国')
    assert 0.0 <= breakdown.segment <= 1.0
    assert 0.0 <= breakdown.features <= 1.0
    assert 0.0 <= breakdown.tone <= config.tone_penalty
    assert 0.0 <= breakdown.overall <= 1.0
