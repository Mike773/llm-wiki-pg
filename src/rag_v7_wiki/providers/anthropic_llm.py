from __future__ import annotations

import os
from typing import TypeVar

import anthropic
import instructor
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AnthropicLLM:
    """Anthropic LLM с structured output через `instructor`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ):
        self._raw = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._instructor = instructor.from_anthropic(self._raw)
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> str:
        message = self._raw.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        return self._instructor.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            response_model=schema,
        )
