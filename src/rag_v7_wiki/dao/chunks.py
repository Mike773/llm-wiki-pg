from __future__ import annotations

from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager


class ChunkDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def bulk_insert(
        self,
        direction_key: str,
        document_id: int,
        chunks: list[tuple[int, str, int]],
    ) -> list[int]:
        """chunks: list of (ord, content, length). Возвращает id'шники в том же порядке."""
        if not chunks:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            ids: list[int] = []
            for ord_, content, length in chunks:
                cur.execute(
                    """
                    INSERT INTO rag_v7.chunks (direction_key, document_id, ord, content, length)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, ord) DO UPDATE
                        SET content = EXCLUDED.content,
                            length = EXCLUDED.length
                    RETURNING id;
                    """,
                    (direction_key, document_id, ord_, content, length),
                )
                ids.append(cur.fetchone()["id"])
            return ids

    def for_document(self, direction_key: str, document_id: int) -> list[dict[str, Any]]:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ord, content, length
                FROM rag_v7.chunks
                WHERE document_id = %s AND direction_key = %s
                ORDER BY ord;
                """,
                (document_id, direction_key),
            )
            return cur.fetchall()
