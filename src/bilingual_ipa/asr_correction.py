"""Utilities for contextual ASR correction using an OpenRouter LLM."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Sequence

import openai

from .conversion import text_to_ipa
from .phonetic_search import PhoneticWindowRetriever, WindowDistance


_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "asr_contextual_correction_prompt.txt"


@dataclass(frozen=True)
class CorrectionCandidate:
    """Candidate correction span provided to the LLM."""

    id: str
    start: int
    end: int
    surface: str
    suggestions: List[str]
    notes: str | None = None


class ASRContextualCorrector:
    """Use OpenRouter to decide on contextual corrections for ASR output."""

    def __init__(
        self,
        *,
        model: str,
        prompt_path: Path | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._model = model
        self._prompt_path = prompt_path or _DEFAULT_PROMPT_PATH
        self._prompt = self._prompt_path.read_text(encoding="utf-8")
        self._client = client

    @staticmethod
    def _token_char_spans(sentence: str, tokens: Sequence[str]) -> list[tuple[int | None, int | None]]:
        spans: list[tuple[int | None, int | None]] = []
        search_start = 0
        for token in tokens:
            if not token:
                spans.append((None, None))
                continue
            index = sentence.find(token, search_start)
            if index == -1:
                spans.append((None, None))
            else:
                end = index + len(token)
                spans.append((index, end))
                search_start = end
        return spans

    @classmethod
    def _build_candidate(
        cls,
        sentence: str,
        spans: Sequence[tuple[int | None, int | None]],
        window: WindowDistance,
        candidate_id: str,
    ) -> CorrectionCandidate | None:
        if window.phrase is None:
            return None

        char_start: int | None = None
        char_end: int | None = None

        for index in range(window.start_index, window.end_index):
            if index < 0 or index >= len(spans):
                continue
            token_start, token_end = spans[index]
            if token_start is None or token_end is None:
                continue
            if char_start is None:
                char_start = token_start
            char_end = token_end

        if char_start is None or char_end is None or char_start >= char_end:
            return None

        surface = sentence[char_start:char_end]
        notes = f"phones={window.phones}; distance={window.distance:.4f}" if window.phones else f"distance={window.distance:.4f}"

        return CorrectionCandidate(
            id=candidate_id,
            start=char_start,
            end=char_end,
            surface=surface,
            suggestions=[window.phrase],
            notes=notes,
        )

    def _build_candidates(
        self,
        sentence: str,
        windows: Sequence[WindowDistance],
    ) -> list[CorrectionCandidate]:
        tokens = text_to_ipa(sentence).tokens
        spans = self._token_char_spans(sentence, tokens)
        candidates: list[CorrectionCandidate] = []
        for idx, window in enumerate(windows):
            candidate = self._build_candidate(sentence, spans, window, f"candidate_{idx}")
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _get_client(self, api_key: str | None) -> openai.OpenAI:
        if self._client is not None:
            return self._client
        if not api_key:
            raise ValueError("An OpenRouter API key must be provided when no client is supplied.")
        return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def correct_sentence(
        self,
        sentence: str,
        retriever: PhoneticWindowRetriever,
        *,
        top_k: int,
        api_key: str | None = None,
    ) -> dict:
        """Use the OpenRouter model to decide on corrections for ``sentence``."""

        windows = retriever.top_k(top_k)
        candidates = self._build_candidates(sentence, windows)
        payload = {
            "sentence": sentence,
            "candidates": [
                {
                    "id": candidate.id,
                    "start": candidate.start,
                    "end": candidate.end,
                    "surface": candidate.surface,
                    "suggestions": candidate.suggestions,
                    **({"notes": candidate.notes} if candidate.notes else {}),
                }
                for candidate in candidates
            ],
        }

        client = self._get_client(api_key)
        response = client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        if not hasattr(response, "output_text"):
            raise RuntimeError("Unexpected response structure from OpenRouter client.")

        return json.loads(response.output_text)


__all__ = ["ASRContextualCorrector", "CorrectionCandidate"]

