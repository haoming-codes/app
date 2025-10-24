"""ASR contextual correction via OpenRouter LLM."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI

from .conversion import IPAConversionResult, text_to_ipa
from .phonetic_search import WindowDistance

PROMPT_PATH = Path(__file__).resolve().parents[2] / "asr_contextual_correction_prompt.txt"


def _token_spans(sentence: str, conversion: IPAConversionResult) -> list[tuple[int, int]]:
    """Return character spans for each token in ``sentence``.

    Args:
        sentence: The original sentence text.
        conversion: IPA conversion result containing ordered tokens.

    Returns:
        A list where each entry is a ``(start, end)`` tuple denoting the
        half-open character span ``[start, end)`` for the corresponding token.

    Raises:
        ValueError: If a token from ``conversion`` cannot be located within the
            ``sentence`` using sequential search.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in conversion.tokens:
        start = sentence.find(token, cursor)
        if start == -1:
            raise ValueError(
                f"Unable to align token '{token}' with the sentence starting at index {cursor}."
            )
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def build_correction_candidates(
    sentence: str, windows: Sequence[WindowDistance]
) -> list[dict[str, Any]]:
    """Construct candidate correction payloads for the contextual LLM.

    Args:
        sentence: Raw ASR sentence to potentially correct.
        windows: Top ``WindowDistance`` results from
            :meth:`~bilingual_ipa.phonetic_search.PhoneticWindowRetriever.top_k`.

    Returns:
        Candidate entries formatted according to the prompt specification.
    """

    if not windows:
        return []

    conversion = text_to_ipa(sentence)
    spans = _token_spans(sentence, conversion)

    candidates: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if window.start_index < 0 or window.end_index > len(spans):
            raise ValueError("Window token indices fall outside the sentence tokens.")
        if window.start_index >= window.end_index:
            raise ValueError("Window must span at least one token to form a candidate.")

        token_start = window.start_index
        token_end = window.end_index - 1
        char_start = spans[token_start][0]
        char_end = spans[token_end][1]
        surface = sentence[char_start:char_end]

        suggestions: list[str] = []
        if window.phrase:
            suggestions.append(window.phrase)

        candidate = {
            "id": f"window_{index}_{window.start_index}_{window.end_index}",
            "start": char_start,
            "end": char_end,
            "surface": surface,
            "suggestions": suggestions,
            "notes": f"distance={window.distance:.4f}; phones={window.phones}",
        }
        candidates.append(candidate)

    return candidates


class ASRContextualCorrector:
    """Client for contextual sentence correction using OpenRouter."""

    def __init__(
        self,
        model: str,
        *,
        client: OpenAI | None = None,
        prompt_path: Path | None = None,
        temperature: float = 0.0,
    ) -> None:
        if prompt_path is None:
            prompt_path = PROMPT_PATH
        self._prompt_path = Path(prompt_path)
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8")

        if client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable must be set to create the client."
                )
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        self._client = client
        self._model = model
        self._temperature = temperature

    @property
    def system_prompt(self) -> str:
        """Return the system prompt text supplied to the LLM."""

        return self._system_prompt

    def correct(
        self, sentence: str, windows: Sequence[WindowDistance]
    ) -> dict[str, Any]:
        """Request contextual corrections for ``sentence`` from the LLM."""

        candidates = build_correction_candidates(sentence, windows)
        payload = {"sentence": sentence, "candidates": candidates}

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self._temperature,
        )

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError("LLM response did not include message content.") from exc

        if not content:
            raise ValueError("LLM response content was empty.")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response content was not valid JSON.") from exc


__all__ = ["ASRContextualCorrector", "build_correction_candidates"]
