from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def to_vec(values: list[float] | None) -> Vector | None:
    """Оборачивает list[float] в pgvector.Vector для корректного дамп-адаптера.

    pgvector регистрирует Dumper только для numpy.ndarray и Vector — поэтому
    plain list нужно явно завернуть, иначе psycopg отправит как double[].
    """
    return None if values is None else Vector(values)


class ConnectionManager:
    """Тонкая обёртка над psycopg ConnectionPool с регистрацией pgvector."""

    def __init__(self, dsn_or_pool: str | ConnectionPool):
        if isinstance(dsn_or_pool, ConnectionPool):
            self._pool = dsn_or_pool
            self._owned = False
        else:
            self._pool = ConnectionPool(
                conninfo=dsn_or_pool,
                min_size=1,
                max_size=10,
                kwargs={"row_factory": dict_row},
                configure=self._configure_connection,
                open=True,
            )
            self._owned = True

    @staticmethod
    def _configure_connection(conn: psycopg.Connection) -> None:
        register_vector(conn)

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as conn:
            register_vector(conn)
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.transaction():
                yield conn

    def close(self) -> None:
        if self._owned:
            self._pool.close()
