from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bilingual_ipa.contextual_correction import (
    ASRContextualCorrector,
    build_correction_candidates,
)
from bilingual_ipa.conversion import IPAConversionResult
from bilingual_ipa.phonetic_search import WindowDistance


@pytest.fixture()
def mock_conversion() -> IPAConversionResult:
    return IPAConversionResult(
        phones=["p1", "p2"],
        tone_marks=["", ""],
        stress_marks=["", ""],
        syllable_counts=[1, 1],
        tokens=["foo", "bar"],
    )


def test_build_correction_candidates_generates_prompt_payload(
    monkeypatch: pytest.MonkeyPatch, mock_conversion: IPAConversionResult
) -> None:
    sentence = "foo baz bar"
    windows = [
        WindowDistance(
            start_index=0,
            end_index=1,
            phones="p1",
            syllable_count=1,
            distance=0.1,
            phrase="foo",
        ),
        WindowDistance(
            start_index=1,
            end_index=2,
            phones="p2",
            syllable_count=1,
            distance=0.2,
            phrase="bar",
        ),
    ]

    monkeypatch.setattr(
        "bilingual_ipa.contextual_correction.text_to_ipa",
        lambda _: mock_conversion,
    )

    candidates = build_correction_candidates(sentence, windows)

    assert candidates == [
        {
            "id": "window_0_0_1",
            "start": 0,
            "end": 3,
            "surface": "foo",
            "suggestions": ["foo"],
            "notes": "distance=0.1000; phones=p1",
        },
        {
            "id": "window_1_1_2",
            "start": 8,
            "end": 11,
            "surface": "bar",
            "suggestions": ["bar"],
            "notes": "distance=0.2000; phones=p2",
        },
    ]


def test_correct_sends_payload_and_parses_json(
    monkeypatch: pytest.MonkeyPatch, mock_conversion: IPAConversionResult
) -> None:
    sentence = "foo baz bar"
    window = WindowDistance(
        start_index=0,
        end_index=1,
        phones="p1",
        syllable_count=1,
        distance=0.1,
        phrase="foo",
    )

    monkeypatch.setattr(
        "bilingual_ipa.contextual_correction.text_to_ipa",
        lambda _: mock_conversion,
    )

    expected_result = {
        "decision": "no_change",
        "corrected_sentence": sentence,
        "replacements": [],
        "skipped_candidates": [],
        "warnings": [],
    }

    mock_create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(expected_result)))]
        )
    )
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)))

    corrector = ASRContextualCorrector(model="openrouter/test", client=mock_client)
    result = corrector.correct(sentence, [window])

    assert result == expected_result

    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "openrouter/test"
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    payload = json.loads(messages[1]["content"])
    assert payload["sentence"] == sentence
    assert payload["candidates"][0]["suggestions"] == ["foo"]


def test_correct_raises_for_invalid_json(monkeypatch: pytest.MonkeyPatch, mock_conversion: IPAConversionResult) -> None:
    sentence = "foo baz bar"
    window = WindowDistance(
        start_index=0,
        end_index=1,
        phones="p1",
        syllable_count=1,
        distance=0.1,
        phrase="foo",
    )

    monkeypatch.setattr(
        "bilingual_ipa.contextual_correction.text_to_ipa",
        lambda _: mock_conversion,
    )

    mock_create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
    )
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)))

    corrector = ASRContextualCorrector(model="openrouter/test", client=mock_client)

    with pytest.raises(ValueError):
        corrector.correct(sentence, [window])
