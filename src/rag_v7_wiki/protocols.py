from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class Embedder(Protocol):
    """Производит эмбеддинги фиксированной размерности.

    Размерность должна совпадать с заложенной в миграции (vector(N) в DDL).
    Дефолтное значение в проекте — 2560.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLM(Protocol):
    """Синхронный LLM-клиент с поддержкой текстовых и structured вызовов."""

    def complete(self, system: str, user: str) -> str: ...

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...

    @property
    def model_name(self) -> str: ...
