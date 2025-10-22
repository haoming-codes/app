import numpy as np

from rag_asr_corrector.ipa import MultilingualIPAConverter


def test_multilingual_ipa_contains_tones_and_stress():
    converter = MultilingualIPAConverter()
    chinese = converter.ipa("美国")
    assert chinese.chinese_char_count == 2
    assert chinese.tones == [3, 2]
    assert chinese.ipa
    assert chinese.segments
    english = converter.ipa("Mini Map")
    assert english.english_word_count == 2
    assert english.stress_pattern == [2, 2]
    assert english.feature_vectors


def test_acronym_is_spelled_out():
    converter = MultilingualIPAConverter()
    result = converter.ipa("AG.AL")
    assert result.stress_pattern == [2, 2]
    assert result.ipa
