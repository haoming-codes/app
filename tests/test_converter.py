from unittest.mock import call, patch

import pytest

from bilingual_ipa import text_to_ipa


def test_text_to_ipa_processes_bilingual_text():
    with patch("bilingual_ipa.converter.phonemize", side_effect=["həˈləʊ", "ni˧˥˩"]) as mock_phonemize:
        result = text_to_ipa("Hello你好")

    assert result == "həˈləʊni˧˥˩"
    assert mock_phonemize.mock_calls == [
        call("Hello", language="en", backend="espeak"),
        call("你好", language="cmn", backend="espeak"),
    ]


def test_text_to_ipa_preserves_non_language_tokens():
    with patch(
        "bilingual_ipa.converter.phonemize",
        side_effect=["həˈləʊ", "ʂɤ˥˩ tɕjɛ˥˩", "ni˧˥˩"],
    ) as mock_phonemize:
        result = text_to_ipa("Hello, 世界!你好")

    assert result == "həˈləʊ, ʂɤ˥˩ tɕjɛ˥˩!ni˧˥˩"
    assert mock_phonemize.mock_calls == [
        call("Hello", language="en", backend="espeak"),
        call("世界", language="cmn", backend="espeak"),
        call("你好", language="cmn", backend="espeak"),
    ]


def test_kwargs_are_forwarded():
    with patch("bilingual_ipa.converter.phonemize", return_value="ipa") as mock_phonemize:
        text_to_ipa("Hi", strip=True)

    mock_phonemize.assert_called_once_with("Hi", language="en", backend="espeak", strip=True)


def test_language_argument_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", language="en")


def test_non_espeak_backend_rejected():
    with pytest.raises(ValueError):
        text_to_ipa("Hi", backend="segments")
