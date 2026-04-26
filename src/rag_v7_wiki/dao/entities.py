from __future__ import annotations

import json
from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager, to_vec


class EntityDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def find_similar(
        self,
        direction_key: str,
        entity_type: str,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Возвращает кандидатов для entity-resolution: id, canonical_name, similarity (cosine)."""
        q = to_vec(query_embedding)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_name, salient_attrs,
                       1 - (canonical_name_embedding <=> %s) AS similarity
                FROM rag_v7.entities
                WHERE direction_key = %s AND entity_type = %s
                ORDER BY canonical_name_embedding <=> %s
                LIMIT %s;
                """,
                (q, direction_key, entity_type, q, top_k),
            )
            rows = cur.fetchall()
            return [r for r in rows if (r.get("similarity") or 0.0) >= threshold]

    def upsert(
        self,
        direction_key: str,
        entity_type: str,
        canonical_name: str,
        canonical_name_embedding: list[float],
        salient_attrs: dict[str, Any] | None = None,
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.entities (
                    direction_key, entity_type, canonical_name,
                    canonical_name_embedding, salient_attrs
                )
                VALUES (%s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                ON CONFLICT (direction_key, entity_type, canonical_name) DO UPDATE
                    SET last_seen_at = now(),
                        salient_attrs = COALESCE(EXCLUDED.salient_attrs, rag_v7.entities.salient_attrs)
                RETURNING id;
                """,
                (
                    direction_key,
                    entity_type,
                    canonical_name,
                    to_vec(canonical_name_embedding),
                    json.dumps(salient_attrs) if salient_attrs else None,
                ),
            )
            return cur.fetchone()["id"]

    def merge_attrs(
        self,
        entity_id: int,
        new_attrs: dict[str, Any],
    ) -> None:
        if not new_attrs:
            return
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.entities
                SET salient_attrs = salient_attrs || %s::jsonb,
                    last_seen_at = now()
                WHERE id = %s;
                """,
                (json.dumps(new_attrs), entity_id),
            )

    def add_alias(
        self,
        direction_key: str,
        entity_id: int,
        alias: str,
        alias_embedding: list[float] | None = None,
        source: str = "extracted",
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.entity_aliases (
                    entity_id, direction_key, alias, alias_embedding, source
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (entity_id, alias) DO NOTHING;
                """,
                (entity_id, direction_key, alias, to_vec(alias_embedding), source),
            )

    def add_mention(
        self,
        direction_key: str,
        entity_id: int,
        chunk_id: int,
        extracted_form: str,
    ) -> bool:
        """Возвращает True, если упоминание добавлено (не дубликат)."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.entity_mentions (
                    entity_id, chunk_id, direction_key, extracted_form
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (entity_id, chunk_id) DO NOTHING
                RETURNING id;
                """,
                (entity_id, chunk_id, direction_key, extracted_form),
            )
            return cur.fetchone() is not None

    def bump_mention_count(self, entity_id: int, by: int = 1) -> None:
        if by <= 0:
            return
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.entities
                SET mention_count = mention_count + %s,
                    last_seen_at = now()
                WHERE id = %s;
                """,
                (by, entity_id),
            )

    def find_by_alias(
        self, direction_key: str, alias: str
    ) -> int | None:
        """Найти entity_id по alias или canonical_name (case-insensitive)."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM rag_v7.entities
                WHERE direction_key = %s AND lower(canonical_name) = lower(%s)
                LIMIT 1;
                """,
                (direction_key, alias),
            )
            row = cur.fetchone()
            if row:
                return row["id"]
            cur.execute(
                """
                SELECT entity_id FROM rag_v7.entity_aliases
                WHERE direction_key = %s AND lower(alias) = lower(%s)
                LIMIT 1;
                """,
                (direction_key, alias),
            )
            row = cur.fetchone()
            return row["entity_id"] if row else None

    def get(self, direction_key: str, entity_id: int) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity_type, canonical_name, salient_attrs, confidence
                FROM rag_v7.entities
                WHERE id = %s AND direction_key = %s;
                """,
                (entity_id, direction_key),
            )
            return cur.fetchone()
