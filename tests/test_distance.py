from asr_corrector import (
    DistanceConfig,
    DistanceWeights,
    SegmentDistanceConfig,
    SegmentMetricConfig,
    compute_distance,
)


def test_distance_breakdown_contains_components():
    config = DistanceConfig(
        segment=SegmentDistanceConfig(
            metrics=[SegmentMetricConfig(name="phonetic_edit_distance", weight=1.0)],
            feature_weight=1.0,
            metrics_weight=1.0,
            feature_distance="l1",
        ),
        weights=DistanceWeights(segment=0.5, tone=0.3, stress=0.2),
    )
    breakdown = compute_distance("minimap", "Mini Map", config=config)
    assert 0 <= breakdown.total <= 1
    assert "phonetic_edit_distance" in breakdown.metrics
    assert breakdown.segment >= 0
    assert breakdown.feature >= 0
    assert breakdown.tone >= 0
    assert breakdown.stress >= 0


def test_distance_detects_tone_difference():
    config = DistanceConfig()
    breakdown = compute_distance("美国", "mango", config=config)
    assert breakdown.tone > 0
