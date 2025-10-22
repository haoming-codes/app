import pytest

from bilingual_ipa.converter import text_to_ipa


@pytest.fixture
def phonemize_mock(monkeypatch):
    calls = []

    def fake_phonemize(text, *, language, backend="espeak-ng", **kwargs):
        calls.append({
            "text": text,
            "language": language,
            "backend": backend,
            "kwargs": kwargs,
        })
        return f"<{language}:{text}>"

    monkeypatch.setattr("bilingual_ipa.converter.phonemize", fake_phonemize)
    return calls


def test_mixed_language_text(phonemize_mock):
    result = text_to_ipa("你好 world", separator=" ")

    assert result == "<cmn:你好> <en:world>"
    assert phonemize_mock == [
        {
            "text": "你好",
            "language": "cmn",
            "backend": "espeak-ng",
            "kwargs": {"separator": " "},
        },
        {
            "text": "world",
            "language": "en",
            "backend": "espeak-ng",
            "kwargs": {"separator": " "},
        },
    ]


def test_backend_can_be_overridden(phonemize_mock):
    text_to_ipa("Hello", backend="espeak")

    assert phonemize_mock == [
        {
            "text": "Hello",
            "language": "en",
            "backend": "espeak",
            "kwargs": {},
        }
    ]


def test_non_language_characters_preserved(phonemize_mock):
    result = text_to_ipa("Hello, 世界!")

    assert result == "<en:Hello>, <cmn:世界>!"
    assert phonemize_mock == [
        {
            "text": "Hello",
            "language": "en",
            "backend": "espeak-ng",
            "kwargs": {},
        },
        {
            "text": "世界",
            "language": "cmn",
            "backend": "espeak-ng",
            "kwargs": {},
        },
    ]
