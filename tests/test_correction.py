from asr_corrector import CorrectionEngine, DistanceConfig


def test_correction_replaces_minimap():
    engine = CorrectionEngine(
        ["Mini Map", "美国"],
        config=DistanceConfig(correction_threshold=0.3, window_radius=1),
    )
    result = engine.correct("拉开minimap查看")
    assert result.text == "拉开Mini Map查看"
    assert result.applied
    assert result.applied[0].original == "minimap"
