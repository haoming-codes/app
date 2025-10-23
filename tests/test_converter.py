from unittest.mock import call, patch

import pytest

from bilingual_ipa import text_to_ipa


def test_text_to_ipa_processes_bilingual_text():
    with (
        patch("bilingual_ipa.converter.english_to_ipa", return_value="həˈləʊ") as mock_english,
        patch("bilingual_ipa.converter.hanzi_to_ipa", return_value="ni˧˥˩") as mock_chinese,
    ):
        result = text_to_ipa("Hello你好")

    assert result == "hələʊni˧˥˩"
    assert mock_english.mock_calls == [call("Hello")]
    assert mock_chinese.mock_calls == [call("你好")]


def test_text_to_ipa_preserves_non_language_tokens():
    with (
        patch("bilingual_ipa.converter.english_to_ipa", return_value="həˈləʊ, ") as mock_english,
        patch(
            "bilingual_ipa.converter.hanzi_to_ipa",
            return_value="ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩",
        ) as mock_chinese,
    ):
        result = text_to_ipa("Hello, 世界!你好")

    assert result == "hələʊ, ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩"
    assert mock_english.mock_calls == [call("Hello, ")]
    assert mock_chinese.mock_calls == [call("世界!你好")]


def test_kwargs_are_rejected():
    with pytest.raises(TypeError):
        text_to_ipa("Hi", strip=True)


def test_consecutive_language_segments_include_spacing_and_punctuation():
    text = "你好。hello world 你好, 你在吗"
    with (
        patch("bilingual_ipa.converter.hanzi_to_ipa", side_effect=["ipa1", "ipa3"]) as mock_chinese,
        patch("bilingual_ipa.converter.english_to_ipa", return_value="ipa2") as mock_english,
    ):
        result = text_to_ipa(text)

    assert result == "ipaipaipa"
    assert mock_chinese.mock_calls == [call("你好。"), call("你好, 你在吗")]
    assert mock_english.mock_calls == [call("hello world ")]


