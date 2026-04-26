from __future__ import annotations

import json
from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager, to_vec


class DocumentDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def ensure_direction(
        self,
        key: str,
        name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.directions (key, name, description, settings)
                VALUES (%s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                ON CONFLICT (key) DO NOTHING;
                """,
                (key, name or key, description, settings),
            )

    def list_pending_ids(self, direction_key: str, limit: int) -> list[int]:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM rag_v7.documents
                WHERE direction_key = %s AND status NOT IN ('processed', 'failed')
                ORDER BY id
                LIMIT %s;
                """,
                (direction_key, limit),
            )
            return [row["id"] for row in cur.fetchall()]

    def get(self, direction_key: str, doc_id: int) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, direction_key, external_id, content, needs_chunking,
                       status, summary, summary_embedding, metadata, redactions
                FROM rag_v7.documents
                WHERE id = %s AND direction_key = %s;
                """,
                (doc_id, direction_key),
            )
            return cur.fetchone()

    def set_status(
        self,
        direction_key: str,
        doc_id: int,
        status: str,
        failed_step: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.documents
                SET status = %s::rag_v7.document_status,
                    failed_step = %s,
                    error = %s,
                    processed_at = CASE WHEN %s = 'processed' THEN now() ELSE processed_at END
                WHERE id = %s AND direction_key = %s;
                """,
                (status, failed_step, error, status, doc_id, direction_key),
            )

    def set_summary(
        self,
        direction_key: str,
        doc_id: int,
        summary: str,
        summary_embedding: list[float] | None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.documents
                SET summary = %s, summary_embedding = %s
                WHERE id = %s AND direction_key = %s;
                """,
                (summary, to_vec(summary_embedding), doc_id, direction_key),
            )

    def set_redacted_content(
        self,
        direction_key: str,
        doc_id: int,
        redacted_content: str,
        redactions: list[dict[str, Any]],
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.documents
                SET content = %s,
                    redactions = %s::jsonb
                WHERE id = %s AND direction_key = %s;
                """,
                (redacted_content, json.dumps(redactions), doc_id, direction_key),
            )
