from unittest.mock import MagicMock, call, patch

import pytest

from bilingual_ipa import text_to_ipa


def test_text_to_ipa_processes_bilingual_text():
    transliterators = {
        "en-us": MagicMock(transliterate=MagicMock(return_value="həˈləʊ")),
        "cmn": MagicMock(transliterate=MagicMock(return_value="ni˧˥˩")),
    }

    with patch("bilingual_ipa.converter._get_transliterator", side_effect=lambda code: transliterators[code]) as mock_get:
        result = text_to_ipa("Hello你好")

    assert result == "həˈləʊni˧˥˩"
    assert mock_get.mock_calls == [call("en-us"), call("cmn")]
    assert transliterators["en-us"].transliterate.mock_calls == [call("Hello")]
    assert transliterators["cmn"].transliterate.mock_calls == [call("你好")]


def test_text_to_ipa_preserves_non_language_tokens():
    transliterators = {
        "en-us": MagicMock(transliterate=MagicMock(return_value="həˈləʊ, ")),
        "cmn": MagicMock(transliterate=MagicMock(return_value="ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩")),
    }

    with patch("bilingual_ipa.converter._get_transliterator", side_effect=lambda code: transliterators[code]) as mock_get:
        result = text_to_ipa("Hello, 世界!你好")

    assert result == "həˈləʊ, ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩"
    assert mock_get.mock_calls == [call("en-us"), call("cmn")]
    assert transliterators["en-us"].transliterate.mock_calls == [call("Hello, ")]
    assert transliterators["cmn"].transliterate.mock_calls == [call("世界!你好")]


def test_unexpected_kwargs_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", strip=True)


def test_consecutive_language_segments_include_spacing_and_punctuation():
    text = "你好。hello world 你好, 你在吗"
    outputs = iter(["ipa1", "ipa2", "ipa3"])

    def fake_get_transliterator(code: str):
        transliterator = MagicMock()
        transliterator.transliterate.side_effect = lambda segment: next(outputs)
        return transliterator

    with patch("bilingual_ipa.converter._get_transliterator", side_effect=fake_get_transliterator) as mock_get:
        result = text_to_ipa(text)

    assert result == "ipaipaipa"
    assert mock_get.mock_calls == [call("cmn"), call("en-us"), call("cmn")]


def test_language_argument_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", language="en")


def test_non_espeak_backend_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", backend="segments")
