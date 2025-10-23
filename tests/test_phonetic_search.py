"""Tests for sliding-window phonetic search utilities."""
from __future__ import annotations

import pytest

from bilingual_ipa import phone_distance, text_to_ipa
from bilingual_ipa.phonetic_search import WindowDistance, window_phonetic_distances


def _distance_for_window(window: WindowDistance, phrase: str) -> float:
    phrase_phone = "".join(text_to_ipa(phrase).phones)
    return phone_distance(window.phones, phrase_phone, metrics="phonetic_edit_distance")


def test_window_phonetic_distances_identifies_english_phrase() -> None:
    sentence = "I have a cat and a dog"
    phrase = "a cat"

    windows = window_phonetic_distances(
        sentence,
        phrase,
        metrics="phonetic_edit_distance",
        syllable_tolerance=0,
    )

    assert windows, "Expected at least one matching window"

    best = min(windows, key=lambda result: result.distance)

    sentence_result = text_to_ipa(sentence)
    expected_phones = "".join(sentence_result.phones[2:4])

    assert best.start_index == 2
    assert best.end_index == 4
    assert best.phones == expected_phones
    assert best.distance == pytest.approx(0.0)
    assert best.distance == pytest.approx(
        _distance_for_window(best, phrase)
    ), "Distance should align with phone_distance helper"


def test_window_phonetic_distances_handles_chinese_phrase() -> None:
    sentence = "你好世界"
    phrase = "世界"

    windows = window_phonetic_distances(
        sentence,
        phrase,
        metrics="phonetic_edit_distance",
        syllable_tolerance=0,
    )

    assert windows, "Expected at least one matching window"

    best = min(windows, key=lambda result: result.distance)

    sentence_result = text_to_ipa(sentence)
    expected_phones = "".join(sentence_result.phones[2:4])

    assert best.start_index == 2
    assert best.end_index == 4
    assert best.phones == expected_phones
    assert best.distance == pytest.approx(0.0)


def test_window_phonetic_distances_requires_phrase_phones() -> None:
    with pytest.raises(ValueError):
        window_phonetic_distances("hello world", "!!!")

