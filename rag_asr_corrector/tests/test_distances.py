from rag_asr_corrector.config import DistanceConfig
from rag_asr_corrector.distances import DistanceCalculator


def test_distance_prefers_correct_pronunciation():
    calculator = DistanceCalculator(DistanceConfig(stress_weight=0.1))
    same = calculator.distance("美国", "美国").total
    confused = calculator.distance("美国", "mango").total
    wrong = calculator.distance("美国", "很好").total
    assert same <= 1e-6
    assert confused < 0.3
    assert wrong < 0.3


def test_distance_between_english_words():
    calculator = DistanceCalculator()
    close = calculator.distance("Mini Map", "minivan").total
    far = calculator.distance("Mini Map", "恒哥").total
    assert close < far
