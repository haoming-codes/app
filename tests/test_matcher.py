from rag_phonetic_correction import (
    LexiconEntry,
    PanphonFeatureDistance,
    PhoneticConverter,
    PhoneticMatcher,
    DistanceCalculator,
    ToneDistance,
)


def build_matcher(threshold: float = 0.6) -> PhoneticMatcher:
    converter = PhoneticConverter()
    calculator = DistanceCalculator(
        segmental=PanphonFeatureDistance(),
        tone=ToneDistance(),
        segment_weight=1.0,
        tone_weight=0.3,
    )
    return PhoneticMatcher(
        converter=converter,
        distance_calculator=calculator,
        default_threshold=threshold,
    )


def test_correct_text_replaces_close_match() -> None:
    matcher = build_matcher()
    lexicon = [LexiconEntry(term="微软")]
    original = "他在美软总部工作"
    corrected, matches = matcher.correct_text(original, lexicon)
    assert corrected == "他在微软总部工作"
    assert matches
    assert matches[0].original == "美软"


def test_no_match_when_distance_above_threshold() -> None:
    matcher = build_matcher(threshold=0.02)
    lexicon = [LexiconEntry(term="阿里巴巴")]
    original = "我们参观了阿狸巴巴的展台"
    corrected, matches = matcher.correct_text(original, lexicon)
    assert corrected == original
    assert not matches
