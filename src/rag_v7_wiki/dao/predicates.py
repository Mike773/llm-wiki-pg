from __future__ import annotations

from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager, to_vec


class PredicateDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def find_similar(
        self,
        direction_key: str,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        q = to_vec(query_embedding)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical, description, times_used,
                       1 - (embedding <=> %s) AS similarity
                FROM rag_v7.canonical_predicates
                WHERE direction_key = %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (q, direction_key, q, top_k),
            )
            rows = cur.fetchall()
            return [r for r in rows if (r.get("similarity") or 0.0) >= threshold]

    def get_by_canonical(
        self, direction_key: str, canonical: str
    ) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical, description, times_used
                FROM rag_v7.canonical_predicates
                WHERE direction_key = %s AND canonical = %s;
                """,
                (direction_key, canonical),
            )
            return cur.fetchone()

    def upsert(
        self,
        direction_key: str,
        canonical: str,
        embedding: list[float],
        description: str | None = None,
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.canonical_predicates (
                    direction_key, canonical, embedding, description
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (direction_key, canonical) DO UPDATE
                    SET description = COALESCE(EXCLUDED.description,
                                               rag_v7.canonical_predicates.description)
                RETURNING id;
                """,
                (direction_key, canonical, to_vec(embedding), description),
            )
            return cur.fetchone()["id"]

    def bump_use(self, predicate_id: int) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.canonical_predicates
                SET times_used = times_used + 1
                WHERE id = %s;
                """,
                (predicate_id,),
            )
