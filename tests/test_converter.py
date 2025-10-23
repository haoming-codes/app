from unittest.mock import call, patch

import pytest

from bilingual_ipa import text_to_ipa


def test_text_to_ipa_processes_bilingual_text():
    with patch("bilingual_ipa.converter._transliterate", side_effect=["həˈləʊ", "ni˧˥˩"]) as mock_transliterate:
        result = text_to_ipa("Hello你好")

    assert result == "hələʊni˧˥˩"
    assert mock_transliterate.mock_calls == [
        call("eng-Latn", "Hello"),
        call("cmn-Hans", "你好"),
    ]


def test_text_to_ipa_preserves_non_language_tokens():
    with patch(
        "bilingual_ipa.converter._transliterate",
        side_effect=["həˈləʊ, ", "ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩"],
    ) as mock_transliterate:
        result = text_to_ipa("Hello, 世界!你好")

    assert result == "hələʊ, ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩"
    assert mock_transliterate.mock_calls == [
        call("eng-Latn", "Hello, "),
        call("cmn-Hans", "世界!你好"),
    ]


def test_kwargs_are_forwarded():
    with patch("bilingual_ipa.converter._transliterate", return_value="ipa") as mock_transliterate:
        text_to_ipa("Hi", strip=True)

    mock_transliterate.assert_called_once_with("eng-Latn", "Hi", strip=True)


def test_consecutive_language_segments_include_spacing_and_punctuation():
    text = "你好。hello world 你好, 你在吗"
    with patch(
        "bilingual_ipa.converter._transliterate",
        side_effect=["ipa1", "ipa2", "ipa3"],
    ) as mock_transliterate:
        result = text_to_ipa(text)

    assert result == "ipaipaipa"
    assert mock_transliterate.mock_calls == [
        call("cmn-Hans", "你好。"),
        call("eng-Latn", "hello world "),
        call("cmn-Hans", "你好, 你在吗"),
    ]


def test_language_argument_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", language="en")
