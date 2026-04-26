from __future__ import annotations

from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager


class WikiLogDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def append(
        self,
        direction_key: str,
        event_kind: str,
        title: str,
        ref_document_id: int | None = None,
        summary: str | None = None,
        affected_pages: list[int] | None = None,
        affected_claims: list[int] | None = None,
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.wiki_log_entries (
                    direction_key, event_kind, title, ref_document_id, summary,
                    affected_pages, affected_claims
                )
                VALUES (%s, %s::rag_v7.log_event_kind, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    direction_key,
                    event_kind,
                    title,
                    ref_document_id,
                    summary,
                    affected_pages or [],
                    affected_claims or [],
                ),
            )
            return cur.fetchone()["id"]

    def list_recent(
        self, direction_key: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, event_kind, title, ref_document_id, summary,
                       affected_pages, affected_claims
                FROM rag_v7.wiki_log_entries
                WHERE direction_key = %s
                ORDER BY ts DESC, id DESC
                LIMIT %s;
                """,
                (direction_key, limit),
            )
            return cur.fetchall()

    def count(self, direction_key: str) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM rag_v7.wiki_log_entries WHERE direction_key = %s;",
                (direction_key,),
            )
            return cur.fetchone()["n"]
