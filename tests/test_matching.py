from phonetic_correction import DistanceConfig, PhoneticMatcher


def test_matcher_finds_close_candidate():
    config = DistanceConfig(threshold=0.7, window_expansion=1)
    matcher = PhoneticMatcher(config)

    matches = matcher.match("we looked at the minivan later", ["Mini Map"])

    assert matches
    assert matches[0].candidate == "Mini Map"
    assert "minivan" in matches[0].window_text
    assert matches[0].distance.total_distance < config.threshold
