import pytest

from bilingual_ipa.conversion import IPAConversionResult
from bilingual_ipa.distances import (
    CompositeDistanceCalculator,
    PhoneDistanceCalculator,
    ToneDistanceCalculator,
)


def _single_phone_result(phone: str, tone: str = "") -> IPAConversionResult:
    return IPAConversionResult(
        phones=[phone],
        tone_marks=[tone],
        stress_marks=[""],
        syllable_counts=[1],
    )


def test_phone_distance_calculator_with_weighted_mean():
    calculator = PhoneDistanceCalculator(
        metrics=["phonetic_edit_distance", "aline"],
        weights={"aline": 2.0},
        aggregate="mean",
    )
    left = _single_phone_result("p")
    right = _single_phone_result("b")

    expected = (
        PhoneDistanceCalculator.AVAILABLE_METRICS["phonetic_edit_distance"]("p", "b") * 1.0
        + PhoneDistanceCalculator.AVAILABLE_METRICS["aline"]("p", "b") * 2.0
    ) / 3.0

    assert calculator.distance(left, right) == pytest.approx(expected)


def test_phone_distance_calculator_supports_other_aggregations():
    metric_names = ["phonetic_edit_distance", "aline"]
    left = _single_phone_result("p")
    right = _single_phone_result("b")
    metric_values = [
        PhoneDistanceCalculator.AVAILABLE_METRICS[name]("p", "b") for name in metric_names
    ]

    sum_calculator = PhoneDistanceCalculator(metrics=metric_names, aggregate="sum")
    assert sum_calculator.distance(left, right) == pytest.approx(sum(metric_values))

    min_calculator = PhoneDistanceCalculator(metrics=metric_names, aggregate="min")
    assert min_calculator.distance(left, right) == pytest.approx(min(metric_values))

    max_calculator = PhoneDistanceCalculator(metrics=metric_names, aggregate="max")
    assert max_calculator.distance(left, right) == pytest.approx(max(metric_values))


def test_phone_distance_calculator_raises_for_zero_weight_mean():
    calculator = PhoneDistanceCalculator(
        metrics=["aline"],
        weights={"aline": 0.0},
        aggregate="mean",
    )

    with pytest.raises(ValueError):
        calculator.distance(_single_phone_result("p"), _single_phone_result("b"))


def test_tone_distance_calculator_matches_identical_sequences():
    calculator = ToneDistanceCalculator()
    left = _single_phone_result("ni", "˧˥˩")
    right = _single_phone_result("ni", "˧˥˩")
    assert calculator.distance(left, right) == pytest.approx(0.0)


def test_tone_distance_calculator_penalizes_missing_tones():
    calculator = ToneDistanceCalculator()
    left = _single_phone_result("ni", "˧˥˩")
    right = _single_phone_result("ni", "")
    assert calculator.distance(left, right) == pytest.approx(1.0)


def test_composite_distance_calculator_combines_distances():
    phone_calculator = PhoneDistanceCalculator(metrics=["phonetic_edit_distance"], aggregate="sum")
    tone_calculator = ToneDistanceCalculator()
    composite = CompositeDistanceCalculator(
        [phone_calculator, tone_calculator],
        aggregate="sum",
    )

    left = _single_phone_result("p", "˥")
    right = _single_phone_result("b", "˩")

    expected = phone_calculator.distance(left, right) + tone_calculator.distance(left, right)
    assert composite.distance(left, right) == pytest.approx(expected)
