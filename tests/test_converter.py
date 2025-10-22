import math

import pytest

from asr_corrector.converter import MultilingualPhoneticConverter


@pytest.fixture(scope="module")
def converter():
    return MultilingualPhoneticConverter()


def test_ipa_contains_expected_segments(converter):
    ipa = converter.ipa("美国")
    assert "m" in ipa and "k" in ipa


def test_acronym_expansion(converter):
    seq = converter.to_sequence("AG.AL")
    assert any(token.language == "en" for token in seq.tokens)
    assert " " in seq.ipa


def test_stress_detection(converter):
    seq = converter.to_sequence("Mini Map")
    assert seq.stress_sequence
    assert seq.stress_sequence[0] in {0, 1, 2}


def test_tone_sequence(converter):
    seq = converter.to_sequence("成哥")
    assert seq.tone_sequence
    assert all(isinstance(t, int) for t in seq.tone_sequence)
