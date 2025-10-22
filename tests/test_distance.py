from asr_corrector.config import PhoneticScoringConfig
from asr_corrector.distance import PhoneticDistanceCalculator


def test_distance_symmetry():
    calculator = PhoneticDistanceCalculator(
        PhoneticScoringConfig(segment_metric="phonetic_edit", segment_weight=0.4, dtw_weight=0.4, tone_weight=0.1, stress_weight=0.1)
    )
    a = "mango"
    b = "美国"
    assert calculator.distance(a, b) == calculator.distance(b, a)


def test_identical_strings_have_zero_distance():
    calculator = PhoneticDistanceCalculator(PhoneticScoringConfig(segment_weight=1.0, dtw_weight=0.0, tone_weight=0.0, stress_weight=0.0))
    score = calculator.distance("Mini Map", "Mini Map")
    assert score == 0.0


def test_component_breakdown():
    calculator = PhoneticDistanceCalculator(
        PhoneticScoringConfig(segment_weight=0.3, dtw_weight=0.3, tone_weight=0.2, stress_weight=0.2)
    )
    comp = calculator.components("Mini Map", "minivan")
    assert set(comp.keys()) == {"segment", "dtw", "tone", "stress"}
    assert all(value >= 0 for value in comp.values())
