"""Tests for sliding-window phonetic search utilities."""
from __future__ import annotations

import pytest

from bilingual_ipa import text_to_ipa
from bilingual_ipa.conversion import IPAConversionResult
from bilingual_ipa.distances import CompositeDistanceCalculator, PhoneDistanceCalculator, ToneDistanceCalculator
from bilingual_ipa.phonetic_search import window_phonetic_distances


def test_window_phonetic_distances_identifies_english_phrase() -> None:
    sentence = "I have a cat and a dog"
    phrase = "a cat"

    calculator = CompositeDistanceCalculator(
        [
            PhoneDistanceCalculator(metrics="phonetic_edit_distance", aggregate="sum"),
            ToneDistanceCalculator(),
        ],
        aggregate="sum",
    )
    windows = window_phonetic_distances(
        sentence,
        phrase,
        distance_calculator=calculator,
        syllable_tolerance=0,
    )

    assert windows, "Expected at least one matching window"

    best = min(windows, key=lambda result: result.distance)

    sentence_result = text_to_ipa(sentence)
    phrase_result = text_to_ipa(phrase)
    expected_phones = "".join(sentence_result.phones[2:4])

    assert best.start_index == 2
    assert best.end_index == 4
    assert best.phones == expected_phones
    window_result = IPAConversionResult(
        phones=list(sentence_result.phones[best.start_index : best.end_index]),
        tone_marks=list(sentence_result.tone_marks[best.start_index : best.end_index]),
        stress_marks=list(sentence_result.stress_marks[best.start_index : best.end_index]),
        syllable_counts=list(sentence_result.syllable_counts[best.start_index : best.end_index]),
    )
    assert best.distance == pytest.approx(
        calculator.distance(window_result, phrase_result)
    ), "Distance should align with the configured calculator"


def test_window_phonetic_distances_handles_chinese_phrase() -> None:
    sentence = "你好世界"
    phrase = "世界"

    calculator = CompositeDistanceCalculator(
        [
            PhoneDistanceCalculator(metrics="phonetic_edit_distance", aggregate="sum"),
            ToneDistanceCalculator(),
        ],
        aggregate="sum",
    )
    windows = window_phonetic_distances(
        sentence,
        phrase,
        distance_calculator=calculator,
        syllable_tolerance=0,
    )

    assert windows, "Expected at least one matching window"

    best = min(windows, key=lambda result: result.distance)

    sentence_result = text_to_ipa(sentence)
    phrase_result = text_to_ipa(phrase)
    expected_phones = "".join(sentence_result.phones[2:4])

    assert best.start_index == 2
    assert best.end_index == 4
    assert best.phones == expected_phones
    window_result = IPAConversionResult(
        phones=list(sentence_result.phones[best.start_index : best.end_index]),
        tone_marks=list(sentence_result.tone_marks[best.start_index : best.end_index]),
        stress_marks=list(sentence_result.stress_marks[best.start_index : best.end_index]),
        syllable_counts=list(sentence_result.syllable_counts[best.start_index : best.end_index]),
    )
    assert best.distance == pytest.approx(
        calculator.distance(window_result, phrase_result)
    )


def test_window_phonetic_distances_requires_phrase_phones() -> None:
    with pytest.raises(ValueError):
        window_phonetic_distances("hello world", "!!!")

