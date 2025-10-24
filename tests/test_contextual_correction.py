from __future__ import annotations

import json

import pytest

from bilingual_ipa.contextual_correction import ASRContextualCorrector
from bilingual_ipa.distances import PhoneDistanceCalculator
from bilingual_ipa.phonetic_search import PhoneticWindowRetriever


class StubLLMClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None
        self.requested_model: str | None = None
        self.requested_temperature: float | None = None

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.messages = list(messages)
        self.requested_model = model
        self.requested_temperature = temperature
        return json.dumps(self.response, ensure_ascii=False)


def test_correct_sentence_builds_candidates_and_uses_prompt() -> None:
    sentence = "你好世界"
    vocabulary = ["世界", "朋友"]
    retriever = PhoneticWindowRetriever(
        distance_calculator=PhoneDistanceCalculator(
            metrics="phonetic_edit_distance",
            aggregate="sum",
        )
    )

    stub = StubLLMClient(
        {
            "decision": "corrected",
            "corrected_sentence": "你好世界",
            "replacements": [],
            "skipped_candidates": [],
            "warnings": [],
        }
    )

    corrector = ASRContextualCorrector(llm_client=stub, retriever=retriever)
    result = corrector.correct_sentence(sentence, vocabulary, top_k=1, model="stub-model", temperature=0.0)

    assert result["decision"] == "corrected"
    assert stub.messages is not None
    assert stub.messages[0]["role"] == "system"
    assert "ASR" in stub.messages[0]["content"]

    user_payload = json.loads(stub.messages[1]["content"])
    assert user_payload["sentence"] == sentence
    assert user_payload["candidates"], "Expected at least one candidate from top-k"
    candidate = user_payload["candidates"][0]
    assert candidate["suggestions"] == ["世界"]
    assert candidate["surface"] in {"世界", "好世界"}
    assert stub.requested_model == "stub-model"
    assert stub.requested_temperature == 0.0


def test_build_candidates_rejects_negative_top_k() -> None:
    stub = StubLLMClient({"decision": "no_change"})
    corrector = ASRContextualCorrector(llm_client=stub)

    with pytest.raises(ValueError):
        corrector.build_candidates("你好", ["你好"], top_k=-1)
