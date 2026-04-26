from __future__ import annotations

import os

from openai import OpenAI


class OpenAIEmbedder:
    """OpenAI text-embedding с поддержкой произвольного `dimensions` (Matryoshka).

    text-embedding-3-large нативно даёт 3072, но позволяет урезать до любого
    значения через параметр dimensions — это удобно, чтобы попасть ровно в
    vector(2560) из миграции.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dim: int = 2560,
    ):
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dim,
        )
        return [d.embedding for d in resp.data]
