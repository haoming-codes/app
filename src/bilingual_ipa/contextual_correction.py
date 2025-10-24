"""Utilities for contextual ASR correction using an OpenRouter-hosted LLM."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .conversion import text_to_ipa
from .distances import PhoneDistanceCalculator
from .phonetic_search import PhoneticWindowRetriever, WindowDistance


@dataclass(frozen=True, slots=True)
class CorrectionCandidate:
    """Represents a potential correction span for the LLM to evaluate."""

    id: str
    start: int
    end: int
    surface: str
    suggestions: tuple[str, ...]
    notes: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "suggestions": list(self.suggestions),
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


class ChatCompletionClient(Protocol):
    """Protocol describing the minimal interface for chat completion clients."""

    def create_chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the textual response from the chat completion endpoint."""


class OpenRouterLLMClient:
    """Simple OpenRouter chat completion client using ``urllib``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-3.5-sonnet",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OpenRouter API key must be provided via argument or OPENROUTER_API_KEY environment variable.")
        self._base_url = base_url.rstrip("/")
        self._default_model = model
        self._timeout = timeout

    def create_chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        request_model = model or self._default_model
        payload: dict[str, object] = {
            "model": request_model,
            "messages": list(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature

        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "bilingual-ipa",
            },
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure handling
            raise RuntimeError(f"OpenRouter request failed with status {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network failure handling
            raise RuntimeError(f"Failed to reach OpenRouter: {exc.reason}") from exc

        data = json.loads(body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenRouter response structure") from exc


def _load_default_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "asr_contextual_correction_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not locate 'asr_contextual_correction_prompt.txt' relative to the project root."
        ) from exc


def _token_char_spans(sentence: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if not token:
            spans.append((cursor, cursor))
            continue
        start = sentence.find(token, cursor)
        if start == -1:
            normalized = token.strip()
            start = sentence.find(normalized, cursor)
        if start == -1:
            raise ValueError(
                "Unable to align token '%s' within the sentence '%s'." % (token, sentence)
            )
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def _window_to_candidate(
    window: WindowDistance,
    *,
    sentence: str,
    spans: Sequence[tuple[int, int]],
    candidate_id: str,
) -> CorrectionCandidate | None:
    if window.phrase is None:
        return None
    if window.end_index > len(spans):
        return None
    start_char = spans[window.start_index][0]
    end_char = spans[window.end_index - 1][1]
    surface = sentence[start_char:end_char]
    notes = f"phonetic_distance={window.distance:.6f}"
    return CorrectionCandidate(
        id=candidate_id,
        start=start_char,
        end=end_char,
        surface=surface,
        suggestions=(window.phrase,),
        notes=notes,
    )


class ASRContextualCorrector:
    """Use phonetic retrieval results plus an OpenRouter LLM to revise ASR text."""

    def __init__(
        self,
        *,
        llm_client: ChatCompletionClient,
        retriever: PhoneticWindowRetriever | None = None,
        prompt: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        if retriever is None:
            retriever = PhoneticWindowRetriever(
                distance_calculator=PhoneDistanceCalculator(
                    metrics="phonetic_edit_distance",
                    aggregate="sum",
                )
            )
        self._retriever = retriever
        self._prompt = prompt or _load_default_prompt()

    @property
    def prompt(self) -> str:
        return self._prompt

    def build_candidates(
        self,
        sentence: str,
        vocabulary: Iterable[str],
        *,
        top_k: int,
    ) -> list[CorrectionCandidate]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        tokens = text_to_ipa(sentence).tokens
        spans = _token_char_spans(sentence, tokens)

        self._retriever.compute_all_distances(sentence, vocabulary)
        windows = self._retriever.top_k(top_k) if top_k else []

        candidates: list[CorrectionCandidate] = []
        for index, window in enumerate(windows):
            candidate = _window_to_candidate(
                window,
                sentence=sentence,
                spans=spans,
                candidate_id=f"cand_{index}",
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def correct_sentence(
        self,
        sentence: str,
        vocabulary: Iterable[str],
        *,
        top_k: int = 5,
        model: str | None = None,
        temperature: float | None = 0.0,
    ) -> dict[str, object]:
        candidates = self.build_candidates(sentence, vocabulary, top_k=top_k)

        payload = {
            "sentence": sentence,
            "candidates": [candidate.to_payload() for candidate in candidates],
        }

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        response_text = self._llm_client.create_chat_completion(
            messages,
            model=model,
            temperature=temperature,
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc


__all__ = [
    "CorrectionCandidate",
    "ChatCompletionClient",
    "OpenRouterLLMClient",
    "ASRContextualCorrector",
]

