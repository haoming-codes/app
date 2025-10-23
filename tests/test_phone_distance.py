import pytest

from bilingual_ipa.phone_distance import (
    AVAILABLE_DISTANCE_METRICS,
    combine_distances,
    compute_distances,
    phone_distance,
)


@pytest.mark.parametrize("metrics", [None, list(AVAILABLE_DISTANCE_METRICS), tuple(AVAILABLE_DISTANCE_METRICS)])
def test_compute_distances_returns_expected_metrics(metrics):
    distances = compute_distances("p", "b", metrics=metrics)
    assert set(distances) == set(AVAILABLE_DISTANCE_METRICS)
    assert distances["phonetic_edit_distance"] == pytest.approx(0.0322580645)
    assert distances["aline"] == pytest.approx(0.2857142857)
    assert distances["feature_edit_distance"] == pytest.approx(0.0416666667)
    assert distances["hamming_feature_edit_distance"] == pytest.approx(0.0416666667)
    assert distances["weighted_feature_edit_distance"] == pytest.approx(0.25)


def test_compute_distances_handles_single_metric_string():
    distances = compute_distances("p", "b", metrics="aline")
    assert set(distances) == {"aline"}
    assert distances["aline"] == pytest.approx(0.2857142857)


def test_phone_distance_with_weighted_mean():
    value = phone_distance(
        "p",
        "b",
        metrics=["phonetic_edit_distance", "aline"],
        weights={"aline": 2.0},
        aggregate="mean",
    )
    assert value == pytest.approx(0.2012288786)


def test_phone_distance_supports_other_aggregations():
    distances = compute_distances("p", "b", metrics=["phonetic_edit_distance", "aline"])
    assert combine_distances(distances, aggregate="sum") == pytest.approx(0.3179723502)
    assert combine_distances(distances, aggregate="min") == pytest.approx(0.0322580645)
    assert combine_distances(distances, aggregate="max") == pytest.approx(0.2857142857)


def test_combine_distances_with_invalid_strategy():
    distances = {"aline": 0.1}
    with pytest.raises(ValueError):
        combine_distances(distances, aggregate="median")


def test_combine_distances_with_zero_weight_mean_error():
    distances = {"aline": 0.1}
    with pytest.raises(ValueError):
        combine_distances(distances, weights={"aline": 0.0}, aggregate="mean")
