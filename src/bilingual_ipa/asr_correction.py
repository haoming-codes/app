"""LLM-assisted contextual correction utilities for ASR output."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

from .conversion import IPAConversionResult, text_to_ipa
from .phonetic_search import PhoneticWindowRetriever, WindowDistance


@dataclass(frozen=True)
class CorrectionCandidate:
    """Candidate replacement suggested by the phonetic retriever."""

    id: str
    start: int
    end: int
    surface: str
    suggestions: list[str]
    notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "suggestions": self.suggestions,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


def _default_prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "asr_contextual_correction_prompt.txt"


def _load_prompt_text(*, prompt_text: str | None, prompt_path: str | Path | None) -> str:
    if prompt_text is not None and prompt_path is not None:
        raise ValueError("Provide either prompt_text or prompt_path, not both.")
    if prompt_text is not None:
        return prompt_text
    if prompt_path is None:
        prompt_path = _default_prompt_path()
    return Path(prompt_path).read_text(encoding="utf-8")


def _token_spans(sentence: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        index = sentence.find(token, cursor)
        if index == -1:
            index = sentence.find(token)
        if index == -1:
            spans.append((-1, -1))
            continue
        start = index
        end = index + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def _candidate_span(
    spans: Sequence[tuple[int, int]],
    start_token: int,
    end_token: int,
) -> tuple[int, int] | None:
    if start_token < 0 or end_token <= start_token:
        return None
    if start_token >= len(spans) or end_token - 1 >= len(spans):
        return None
    start_span = spans[start_token]
    end_span = spans[end_token - 1]
    if start_span[0] < 0 or end_span[1] < 0:
        return None
    return start_span[0], end_span[1]


class ASRContextualCorrector:
    """Use an OpenRouter-hosted LLM to review phonetic correction candidates."""

    def __init__(
        self,
        *,
        retriever: PhoneticWindowRetriever | None = None,
        client: OpenAI | None = None,
        model: str = "openrouter/auto",
        prompt_text: str | None = None,
        prompt_path: str | Path | None = None,
        default_top_k: int = 5,
    ) -> None:
        self._retriever = retriever or PhoneticWindowRetriever()
        self._client = client or OpenAI()
        self._model = model
        self._prompt = _load_prompt_text(prompt_text=prompt_text, prompt_path=prompt_path)
        self._default_top_k = default_top_k

    @property
    def prompt(self) -> str:
        return self._prompt

    def _build_candidates(
        self,
        sentence: str,
        *,
        windows: Sequence[WindowDistance],
        conversion_result: IPAConversionResult,
    ) -> list[CorrectionCandidate]:
        tokens = conversion_result.tokens
        spans = _token_spans(sentence, tokens)
        candidates: list[CorrectionCandidate] = []

        for index, window in enumerate(windows):
            if window.phrase is None:
                continue
            span_tokens = tokens[window.start_index : window.end_index]
            if not span_tokens:
                continue
            char_span = _candidate_span(spans, window.start_index, window.end_index)
            notes = [f"distance={window.distance:.4f}"]
            if char_span is None:
                surface = "".join(span_tokens)
                notes.append("span_alignment_failed")
                start = end = -1
            else:
                start, end = char_span
                surface = sentence[start:end]
            candidates.append(
                CorrectionCandidate(
                    id=f"cand_{index+1}",
                    start=start,
                    end=end,
                    surface=surface,
                    suggestions=[window.phrase],
                    notes=";".join(notes),
                )
            )
        return candidates

    def gather_candidates(
        self,
        sentence: str,
        vocabulary: Iterable[str],
        *,
        top_k: int | None = None,
    ) -> list[CorrectionCandidate]:
        conversion_result = text_to_ipa(sentence)
        self._retriever.compute_all_distances(sentence, vocabulary)
        k = self._default_top_k if top_k is None else top_k
        windows = self._retriever.top_k(k)
        return self._build_candidates(
            sentence,
            windows=windows,
            conversion_result=conversion_result,
        )

    def correct_sentence(
        self,
        sentence: str,
        vocabulary: Iterable[str],
        *,
        top_k: int | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidates = self.gather_candidates(sentence, vocabulary, top_k=top_k)
        payload: dict[str, Any] = {
            "sentence": sentence,
            "candidates": [candidate.to_payload() for candidate in candidates],
        }
        if extra_payload:
            payload.update(extra_payload)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        try:
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            raise ValueError("LLM response missing content") from exc
        if content is None:
            raise ValueError("LLM response did not contain any content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc


__all__ = ["ASRContextualCorrector", "CorrectionCandidate"]
