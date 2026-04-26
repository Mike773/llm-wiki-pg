from __future__ import annotations

import os

import voyageai


class VoyageEmbedder:
    """Эмбеддер на базе Voyage AI.

    Учтите: воображаемая размерность 2560 (как заложена в миграции по умолчанию)
    не покрывается стандартными моделями Voyage (voyage-3 = 1024, voyage-3-large
    до 2048). Если в вашем проекте dim=2560, используйте свою имплементацию
    Embedder (например, sentence-transformers с подходящей моделью). Этот
    провайдер оставлен как референсная реализация интерфейса.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3-large",
        dim: int = 1024,
        input_type: str = "document",
    ):
        self._client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self._model = model
        self._dim = dim
        self._input_type = input_type

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=self._input_type,
        )
        return result.embeddings
