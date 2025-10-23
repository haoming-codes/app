from unittest.mock import call, patch

import pytest

from bilingual_ipa import text_to_ipa


def test_text_to_ipa_processes_bilingual_text():
    with (
        patch("bilingual_ipa.conversion.english_to_ipa", return_value="həˈləʊ") as mock_english,
        patch("bilingual_ipa.conversion.hanzi_to_ipa", return_value="ni˧˥˩") as mock_chinese,
    ):
        result = text_to_ipa("Hello你好")

    assert result.phones == ["hələʊ", "ni", "ni"]
    assert result.tone_marks == ["", "˧˥˩", "˧˥˩"]
    assert result.stress_marks == ["ˈ", "", ""]
    assert result.syllable_counts == [2, 1, 1]
    assert mock_english.mock_calls == [call("Hello", keep_punct=False)]
    assert mock_chinese.mock_calls == [call("你", delimiter=""), call("好", delimiter="")]


def test_text_to_ipa_preserves_non_language_tokens():
    with (
        patch("bilingual_ipa.conversion.english_to_ipa", side_effect=["həˈləʊ"]) as mock_english,
        patch(
            "bilingual_ipa.conversion.hanzi_to_ipa",
            side_effect=["ʂɤ˥˩", "tɕjɛ˥˩", "ni˧˥˩", "xɑʊ˥˩"],
        ) as mock_chinese,
    ):
        result = text_to_ipa("Hello, 世界!你好")

    assert result.phones == ["hələʊ", "ʂɤ", "tɕjɛ", "ni", "xɑʊ"]
    assert result.tone_marks == ["", "˥˩", "˥˩", "˧˥˩", "˥˩"]
    assert result.stress_marks == ["ˈ", "", "", "", ""]
    assert result.syllable_counts == [2, 1, 1, 1, 1]
    assert mock_english.mock_calls == [call("Hello", keep_punct=False)]
    assert mock_chinese.mock_calls == [
        call("世", delimiter=""),
        call("界", delimiter=""),
        call("你", delimiter=""),
        call("好", delimiter=""),
    ]


def test_kwargs_are_rejected():
    with pytest.raises(TypeError):
        text_to_ipa("Hi", strip=True)


def test_consecutive_language_segments_include_spacing_and_punctuation():
    text = "你好。hello world 你好, 你在吗"
    with (
        patch(
            "bilingual_ipa.conversion.hanzi_to_ipa",
            side_effect=[
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c6",
                "c7",
            ],
        ) as mock_chinese,
        patch("bilingual_ipa.conversion.english_to_ipa", side_effect=["e1", "e2"]) as mock_english,
    ):
        result = text_to_ipa(text)

    assert result.phones == ["c1", "c2", "e1", "e2", "c3", "c4", "c5", "c6", "c7"]
    assert result.syllable_counts == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert mock_chinese.mock_calls == [
        call("你", delimiter=""),
        call("好", delimiter=""),
        call("你", delimiter=""),
        call("好", delimiter=""),
        call("你", delimiter=""),
        call("在", delimiter=""),
        call("吗", delimiter=""),
    ]
    assert mock_english.mock_calls == [call("hello", keep_punct=False), call("world", keep_punct=False)]


def test_all_caps_words_are_split_before_conversion():
    with patch("bilingual_ipa.conversion.english_to_ipa", return_value="ipa") as mock_english:
        result = text_to_ipa("AP")

    assert result.phones == ["ipa", "ipa"]
    assert mock_english.mock_calls == [call("A", keep_punct=False), call("P", keep_punct=False)]


def test_punctuation_is_followed_by_space_before_conversion():
    with patch("bilingual_ipa.conversion.english_to_ipa", return_value="ipa") as mock_english:
        text_to_ipa("Hello.World")

    assert mock_english.mock_calls == [call("Hello", keep_punct=False), call("World", keep_punct=False)]


