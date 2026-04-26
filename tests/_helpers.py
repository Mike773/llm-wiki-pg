from __future__ import annotations

from typing import Any

import psycopg


def insert_document(
    dsn: str,
    direction_key: str,
    content: str,
    needs_chunking: bool = False,
    external_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.execute(
            """
            INSERT INTO rag_v7.documents (
                direction_key, external_id, content, needs_chunking, metadata
            )
            VALUES (%s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
            RETURNING id;
            """,
            (direction_key, external_id, content, needs_chunking, metadata),
        )
        return cur.fetchone()[0]


def count(dsn: str, sql: str, params: tuple = ()) -> int:
    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.execute(sql, params)
        return cur.fetchone()[0]
