from __future__ import annotations

import os
from typing import TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OpenAILLM:
    """OpenAI LLM с structured output через `instructor`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
    ):
        self._raw = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._instructor = instructor.from_openai(self._raw)
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> str:
        resp = self._raw.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        return self._instructor.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=schema,
        )
