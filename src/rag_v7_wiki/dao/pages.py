from __future__ import annotations

import json
from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager, to_vec


class PageDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def get_by_entity(
        self, direction_key: str, entity_id: int
    ) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, title, content_md, version, last_synthesized_at,
                       page_kind, quality_score
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s AND entity_id = %s;
                """,
                (direction_key, entity_id),
            )
            return cur.fetchone()

    def get_singleton(
        self, direction_key: str, page_kind: str
    ) -> dict[str, Any] | None:
        """Получить singleton-страницу (index/log/overview) направления."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, title, content_md, version, last_synthesized_at,
                       page_kind
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s AND page_kind = %s::rag_v7.wiki_page_kind;
                """,
                (direction_key, page_kind),
            )
            return cur.fetchone()

    def get_source_page(
        self, direction_key: str, source_document_id: int
    ) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, title, content_md, version, last_synthesized_at,
                       page_kind, source_document_id
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s
                  AND source_document_id = %s
                  AND page_kind = 'source';
                """,
                (direction_key, source_document_id),
            )
            return cur.fetchone()

    def upsert(
        self,
        direction_key: str,
        entity_id: int,
        slug: str,
        title: str,
        content_md: str,
        content_embedding: list[float],
    ) -> tuple[int, int]:
        """Upsert entity-wiki-страницы. Возвращает (page_id, new_version)."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.wiki_pages (
                    direction_key, entity_id, page_kind, slug, title, content_md,
                    content_embedding, version, last_synthesized_at
                )
                VALUES (%s, %s, 'entity', %s, %s, %s, %s, 1, now())
                ON CONFLICT (direction_key, slug) DO UPDATE
                    SET entity_id = EXCLUDED.entity_id,
                        title = EXCLUDED.title,
                        content_md = EXCLUDED.content_md,
                        content_embedding = EXCLUDED.content_embedding,
                        version = rag_v7.wiki_pages.version + 1,
                        last_synthesized_at = now()
                RETURNING id, version;
                """,
                (
                    direction_key,
                    entity_id,
                    slug,
                    title,
                    content_md,
                    to_vec(content_embedding),
                ),
            )
            row = cur.fetchone()
            return row["id"], row["version"]

    def upsert_source_page(
        self,
        direction_key: str,
        source_document_id: int,
        slug: str,
        title: str,
        content_md: str,
        content_embedding: list[float],
    ) -> tuple[int, int]:
        """Upsert страницы-источника. Уникальность через wiki_pages_dir_source_uniq."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.wiki_pages (
                    direction_key, page_kind, source_document_id, slug, title,
                    content_md, content_embedding, version, last_synthesized_at
                )
                VALUES (%s, 'source', %s, %s, %s, %s, %s, 1, now())
                ON CONFLICT (direction_key, slug) DO UPDATE
                    SET source_document_id = EXCLUDED.source_document_id,
                        title = EXCLUDED.title,
                        content_md = EXCLUDED.content_md,
                        content_embedding = EXCLUDED.content_embedding,
                        version = rag_v7.wiki_pages.version + 1,
                        last_synthesized_at = now()
                RETURNING id, version;
                """,
                (
                    direction_key,
                    source_document_id,
                    slug,
                    title,
                    content_md,
                    to_vec(content_embedding),
                ),
            )
            row = cur.fetchone()
            return row["id"], row["version"]

    def upsert_singleton_page(
        self,
        direction_key: str,
        page_kind: str,
        slug: str,
        title: str,
        content_md: str,
        content_embedding: list[float],
    ) -> tuple[int, int]:
        """Upsert одной из singleton-страниц направления (index/log/overview).

        Уникальность гарантируется wiki_pages_dir_singleton_uniq, поэтому
        ON CONFLICT идёт по (direction_key, page_kind) через явный поиск.
        """
        existing = self.get_singleton(direction_key, page_kind)
        with self._cm.conn() as conn, conn.cursor() as cur:
            if existing is not None:
                cur.execute(
                    """
                    UPDATE rag_v7.wiki_pages
                    SET slug = %s,
                        title = %s,
                        content_md = %s,
                        content_embedding = %s,
                        version = version + 1,
                        last_synthesized_at = now()
                    WHERE id = %s
                    RETURNING id, version;
                    """,
                    (slug, title, content_md, to_vec(content_embedding), existing["id"]),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO rag_v7.wiki_pages (
                        direction_key, page_kind, slug, title, content_md,
                        content_embedding, version, last_synthesized_at
                    )
                    VALUES (%s, %s::rag_v7.wiki_page_kind, %s, %s, %s, %s, 1, now())
                    RETURNING id, version;
                    """,
                    (
                        direction_key,
                        page_kind,
                        slug,
                        title,
                        content_md,
                        to_vec(content_embedding),
                    ),
                )
            row = cur.fetchone()
            return row["id"], row["version"]

    def save_revision(
        self,
        page_id: int,
        version: int,
        content_md: str,
        synthesized_from_claim_ids: list[int],
        llm_model: str | None = None,
        quality_score: float | None = None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.wiki_page_revisions (
                    page_id, version, content_md, synthesized_from_claim_ids,
                    llm_model, quality_score
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (page_id, version) DO NOTHING;
                """,
                (
                    page_id,
                    version,
                    content_md,
                    synthesized_from_claim_ids,
                    llm_model,
                    quality_score,
                ),
            )

    def set_metrics(
        self,
        page_id: int,
        quality_score: float | None,
        coverage_claims: int,
        coverage_unresolved_links: int,
        coverage_contradictions: int,
        body_meta: dict[str, Any] | None = None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.wiki_pages
                SET quality_score = %s,
                    coverage_claims = %s,
                    coverage_unresolved_links = %s,
                    coverage_contradictions = %s,
                    body_meta = COALESCE(%s::jsonb, body_meta)
                WHERE id = %s;
                """,
                (
                    quality_score,
                    coverage_claims,
                    coverage_unresolved_links,
                    coverage_contradictions,
                    json.dumps(body_meta) if body_meta is not None else None,
                    page_id,
                ),
            )

    def upsert_provenance(
        self,
        direction_key: str,
        page_id: int,
        document_counts: dict[int, int],
    ) -> None:
        if not document_counts:
            return
        with self._cm.conn() as conn, conn.cursor() as cur:
            for document_id, count in document_counts.items():
                cur.execute(
                    """
                    INSERT INTO rag_v7.page_sources (
                        page_id, document_id, direction_key, claim_count, last_seen_at
                    )
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (page_id, document_id) DO UPDATE
                        SET claim_count = EXCLUDED.claim_count,
                            last_seen_at = now();
                    """,
                    (page_id, document_id, direction_key, count),
                )

    def list_for_index(self, direction_key: str) -> list[dict[str, Any]]:
        """Список страниц для построения index.md (без index/log самих)."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.page_kind::text AS page_kind, p.slug, p.title,
                       p.last_synthesized_at, p.quality_score, p.coverage_claims,
                       p.coverage_contradictions, e.canonical_name AS entity_name,
                       e.entity_type
                FROM rag_v7.wiki_pages p
                LEFT JOIN rag_v7.entities e ON e.id = p.entity_id
                WHERE p.direction_key = %s
                  AND p.page_kind NOT IN ('index', 'log')
                ORDER BY p.page_kind, p.title;
                """,
                (direction_key,),
            )
            return cur.fetchall()

    def upsert_link(
        self,
        direction_key: str,
        from_page_id: int,
        anchor_text: str,
        to_entity_id: int | None,
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.page_links (
                    direction_key, from_page_id, to_entity_id, anchor_text, resolved
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (from_page_id, anchor_text) DO UPDATE
                    SET to_entity_id = EXCLUDED.to_entity_id,
                        resolved = EXCLUDED.resolved;
                """,
                (
                    direction_key,
                    from_page_id,
                    to_entity_id,
                    anchor_text,
                    to_entity_id is not None,
                ),
            )

    def clear_links(self, from_page_id: int) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rag_v7.page_links WHERE from_page_id = %s;",
                (from_page_id,),
            )

    def count_unresolved_links(self, page_id: int) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS n FROM rag_v7.page_links
                WHERE from_page_id = %s AND resolved = false;
                """,
                (page_id,),
            )
            return cur.fetchone()["n"]

    def list_by_ids(
        self, direction_key: str, page_ids: list[int]
    ) -> list[dict[str, Any]]:
        if not page_ids:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity_id, slug, title, content_md, page_kind
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s AND id = ANY(%s);
                """,
                (direction_key, page_ids),
            )
            return cur.fetchall()
