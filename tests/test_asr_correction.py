"""Tests for contextual ASR correction using OpenRouter."""
from __future__ import annotations

import json

import pytest

from bilingual_ipa.asr_correction import ASRContextualCorrector
from bilingual_ipa.phonetic_search import PhoneticWindowRetriever, WindowDistance


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class DummyResponses:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        return DummyResponse(self._text)


class DummyClient:
    def __init__(self, text: str) -> None:
        self.responses = DummyResponses(text)


def test_correct_sentence_builds_openrouter_payload():
    sentence = "Hello world"
    retriever = PhoneticWindowRetriever()
    retriever._results = [  # type: ignore[attr-defined]
        WindowDistance(
            start_index=0,
            end_index=1,
            phones="həloʊ",
            syllable_count=2,
            distance=0.42,
            phrase="Halo",
        ),
    ]

    client = DummyClient('{"decision": "no_change"}')
    corrector = ASRContextualCorrector(model="test-model", client=client)

    result = corrector.correct_sentence(sentence, retriever, top_k=1)

    assert result == {"decision": "no_change"}

    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == "test-model"

    messages = call["input"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("SYSTEM ROLE")

    payload = json.loads(messages[1]["content"])
    assert payload["sentence"] == sentence
    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["surface"] == "Hello"
    assert candidate["start"] == 0
    assert candidate["end"] == 5
    assert candidate["suggestions"] == ["Halo"]
    assert "distance=0.4200" in candidate["notes"]


def test_correct_sentence_requires_api_key_when_no_client():
    sentence = "Sample"
    retriever = PhoneticWindowRetriever()
    corrector = ASRContextualCorrector(model="test-model")

    with pytest.raises(ValueError):
        corrector.correct_sentence(sentence, retriever, top_k=1)

