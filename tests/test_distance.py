import pytest

from asr_correction.config import DistanceComputationConfig, SegmentalMetricConfig, ToneDistanceConfig
from asr_correction.distance import compute_distance


def make_basic_config():
    metric = SegmentalMetricConfig(name="phonetic_edit_distance", weight=1.0)
    tone = ToneDistanceConfig(weight=1.0, confusion_penalties={("3", "4"): 0.3})
    return DistanceComputationConfig(
        segmental_metrics=[metric],
        tone=tone,
        tradeoff_lambda=0.8,
        threshold=0.5,
    )


def test_distance_symmetry():
    config = make_basic_config()
    a = "阿里巴巴"
    b = "阿里爸爸"
    dist_ab = compute_distance(a, b, config)
    dist_ba = compute_distance(b, a, config)
    assert pytest.approx(dist_ab, rel=1e-6) == dist_ba


def test_distance_lower_for_correct_pronunciation():
    config = make_basic_config()
    correct = "阿里巴巴"
    incorrect = "阿里爸爸"
    unrelated = "谷歌"
    dist_correct = compute_distance(correct, correct, config)
    dist_incorrect = compute_distance(correct, incorrect, config)
    dist_unrelated = compute_distance(correct, unrelated, config)
    assert dist_correct <= dist_incorrect <= dist_unrelated
