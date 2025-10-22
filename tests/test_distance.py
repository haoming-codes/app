from ragasr.config import DistanceConfig
from ragasr.distance import PhoneticDistanceCalculator


def test_distance_prefers_correct_entity():
    config = DistanceConfig(threshold=0.15)
    calc = PhoneticDistanceCalculator(config)
    same = calc.combined_distance("Mini Map", "Mini Map")
    near = calc.combined_distance("minivan", "Mini Map")
    far = calc.combined_distance("美国", "Mini Map")
    assert same <= near < far


def test_distance_components_with_config():
    config = DistanceConfig(segment_metrics=("phonetic",), use_feature_dtw=False, segment_weight=1.0, tone_weight=0.0, stress_weight=0.0)
    calc = PhoneticDistanceCalculator(config)
    result = calc.distance("Mini Map", "minivan")
    assert 0 <= result.segment <= 1
    assert result.tone == 0
    assert result.stress == 0
