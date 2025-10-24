import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bilingual_ipa.asr_correction import ASRContextualCorrector, CorrectionCandidate
from bilingual_ipa.phonetic_search import WindowDistance


class StubRetriever:
    def __init__(self, results: list[WindowDistance]) -> None:
        self._results = results
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def compute_all_distances(self, sentence: str, vocabulary):
        self.calls.append((sentence, tuple(vocabulary)))
        return self._results

    def top_k(self, k: int):
        return self._results[:k]


def test_gather_candidates_returns_span_aligned_entries():
    sentence = "你好 world"
    windows = [
        WindowDistance(
            start_index=0,
            end_index=2,
            phones="",  # unused
            syllable_count=0,
            distance=0.12,
            phrase="你好吗",
        )
    ]
    retriever = StubRetriever(windows)

    conversion_result = SimpleNamespace(
        tokens=["你", "好", "world"],
    )

    with patch("bilingual_ipa.asr_correction.text_to_ipa", return_value=conversion_result):
        corrector = ASRContextualCorrector(
            retriever=retriever,
            client=MagicMock(),
            prompt_text="PROMPT",
        )
        candidates = corrector.gather_candidates(sentence, ["你好吗"], top_k=1)

    assert candidates == [
        CorrectionCandidate(
            id="cand_1",
            start=0,
            end=2,
            surface="你好",
            suggestions=["你好吗"],
            notes="distance=0.1200",
        )
    ]


def test_correct_sentence_invokes_openai_chat_completion_and_parses_json():
    sentence = "test"
    windows = [
        WindowDistance(
            start_index=0,
            end_index=1,
            phones="",
            syllable_count=1,
            distance=0.5,
            phrase="demo",
        )
    ]
    retriever = StubRetriever(windows)

    conversion_result = SimpleNamespace(tokens=["test"])
    with patch("bilingual_ipa.asr_correction.text_to_ipa", return_value=conversion_result):
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision": "no_change"}'))]
        )
        client = SimpleNamespace(chat=mock_chat)

        corrector = ASRContextualCorrector(
            retriever=retriever,
            client=client,  # type: ignore[arg-type]
            prompt_text="PROMPT",
            model="openrouter/unit-test",
        )

        result = corrector.correct_sentence(sentence, ["demo"], top_k=1)

    assert result == {"decision": "no_change"}

    mock_chat.completions.create.assert_called_once()
    call_kwargs = mock_chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openrouter/unit-test"
    assert call_kwargs["messages"][0] == {"role": "system", "content": "PROMPT"}

    user_payload = call_kwargs["messages"][1]["content"]
    payload = json.loads(user_payload)
    assert payload["sentence"] == sentence
    assert payload["candidates"][0]["suggestions"] == ["demo"]


def test_invalid_json_response_raises_value_error():
    retriever = StubRetriever([])
    with patch(
        "bilingual_ipa.asr_correction.text_to_ipa",
        return_value=SimpleNamespace(tokens=[]),
    ):
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        client = SimpleNamespace(chat=mock_chat)

        corrector = ASRContextualCorrector(
            retriever=retriever,
            client=client,  # type: ignore[arg-type]
            prompt_text="PROMPT",
        )

        with pytest.raises(ValueError):
            corrector.correct_sentence("sentence", ["vocab"], top_k=0)
