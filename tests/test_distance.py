import math

from asr_corrector import CorrectionConfig, DistanceCalculator, MetricConfig, ToneConfig


def test_distance_breakdown_metrics_and_tone():
    config = CorrectionConfig(
        metrics=[
            MetricConfig(name="panphon", weight=0.6),
            MetricConfig(name="phonetic_edit", weight=0.4),
        ],
        tone=ToneConfig(weight=1.0, default_cost=1.0),
        segment_lambda=0.5,
        tone_lambda=0.5,
    )
    calculator = DistanceCalculator(config)
    breakdown = calculator.distance("清华大学", "清华大學")
    assert math.isclose(breakdown.segmental, 0.0)
    assert math.isclose(breakdown.tone, 0.0)
    assert math.isclose(breakdown.total, 0.0)
    assert set(breakdown.per_metric) == {"panphon", "phonetic_edit"}


def test_tone_confusion_costs_reduce_distance():
    base_config = CorrectionConfig(
        metrics=[MetricConfig(name="panphon", weight=1.0)],
        tone=ToneConfig(weight=1.0, default_cost=1.0),
    )
    conf_config = CorrectionConfig(
        metrics=[MetricConfig(name="panphon", weight=1.0)],
        tone=ToneConfig(
            weight=1.0,
            default_cost=1.0,
            confusion_costs={"1": {"2": 0.25}, "2": {"1": 0.25}},
        ),
    )
    baseline = DistanceCalculator(base_config).distance("妈", "麻")
    confused = DistanceCalculator(conf_config).distance("妈", "麻")
    assert confused.tone < baseline.tone
    assert confused.total < baseline.total
