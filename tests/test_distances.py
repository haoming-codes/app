import pytest

from phonetic_correction import DistanceConfig, PhoneticDistanceCalculator
from phonetic_correction.phonetics import MultilingualPhonemizer


def test_identical_strings_have_zero_distance():
    config = DistanceConfig()
    calculator = PhoneticDistanceCalculator(config)
    phonemizer = MultilingualPhonemizer()

    first = phonemizer.transcribe("美国")
    second = phonemizer.transcribe("美国")

    result = calculator.distance(first, second)

    assert pytest.approx(result.segment_distance, abs=1e-6) == 0.0
    assert pytest.approx(result.tone_distance, abs=1e-6) == 0.0
    assert pytest.approx(result.stress_distance, abs=1e-6) == 0.0
    assert pytest.approx(result.total_distance, abs=1e-6) == 0.0


def test_tone_mismatch_increases_distance():
    config = DistanceConfig(lambda_segment=0.3, lambda_tone=0.4, lambda_stress=0.3)
    calculator = PhoneticDistanceCalculator(config)
    phonemizer = MultilingualPhonemizer()

    base = phonemizer.transcribe("美国")
    different = phonemizer.transcribe("妹国")

    result = calculator.distance(base, different)

    assert result.tone_distance > 0
    assert result.total_distance > 0
    assert result.total_distance >= result.tone_distance * calculator.config.normalized_lambdas()["tone"]
