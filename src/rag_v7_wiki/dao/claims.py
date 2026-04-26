from __future__ import annotations

from typing import Any

from rag_v7_wiki.dao.connection import ConnectionManager, to_vec


class ClaimDAO:
    def __init__(self, cm: ConnectionManager):
        self._cm = cm

    def find_similar_for_subject(
        self,
        direction_key: str,
        subject_entity_id: int,
        predicate: str,
        claim_embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Кандидаты на supersession/dedup в рамках того же subject+predicate."""
        q = to_vec(claim_embedding)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.predicate, c.object_kind, c.object_entity_id,
                       c.object_text, c.claim_text, c.status, c.confidence,
                       c.times_confirmed, c.first_seen_at, c.last_confirmed_at,
                       1 - (c.claim_embedding <=> %s) AS similarity,
                       e.canonical_name AS object_canonical_name
                FROM rag_v7.claims c
                LEFT JOIN rag_v7.entities e ON e.id = c.object_entity_id
                WHERE c.direction_key = %s
                  AND c.subject_entity_id = %s
                  AND c.predicate = %s
                  AND c.status IN ('active', 'flagged_contradiction')
                ORDER BY c.claim_embedding <=> %s
                LIMIT %s;
                """,
                (q, direction_key, subject_entity_id, predicate, q, top_k),
            )
            rows = cur.fetchall()
            return [r for r in rows if (r.get("similarity") or 0.0) >= threshold]

    def get(self, claim_id: int) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.direction_key, c.subject_entity_id, c.predicate,
                       c.canonical_predicate_id, c.object_kind, c.object_entity_id,
                       c.object_text, c.claim_text, c.confidence, c.times_confirmed,
                       c.tier, c.first_seen_at, c.last_confirmed_at, c.status,
                       c.superseded_by_id,
                       e.canonical_name AS object_canonical_name
                FROM rag_v7.claims c
                LEFT JOIN rag_v7.entities e ON e.id = c.object_entity_id
                WHERE c.id = %s;
                """,
                (claim_id,),
            )
            return cur.fetchone()

    def insert(
        self,
        direction_key: str,
        subject_entity_id: int,
        predicate: str,
        object_kind: str,
        object_entity_id: int | None,
        object_text: str | None,
        claim_text: str,
        claim_embedding: list[float],
        confidence: float,
        canonical_predicate_id: int | None = None,
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.claims (
                    direction_key, subject_entity_id, predicate, canonical_predicate_id,
                    object_kind, object_entity_id, object_text, claim_text,
                    claim_embedding, confidence
                )
                VALUES (%s, %s, %s, %s, %s::rag_v7.claim_object_kind,
                        %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    direction_key,
                    subject_entity_id,
                    predicate,
                    canonical_predicate_id,
                    object_kind,
                    object_entity_id,
                    object_text,
                    claim_text,
                    to_vec(claim_embedding),
                    confidence,
                ),
            )
            return cur.fetchone()["id"]

    def add_citations(
        self,
        direction_key: str,
        claim_id: int,
        chunk_ids: list[int],
    ) -> None:
        if not chunk_ids:
            return
        with self._cm.conn() as conn, conn.cursor() as cur:
            for chunk_id in chunk_ids:
                cur.execute(
                    """
                    INSERT INTO rag_v7.claim_citations (claim_id, chunk_id, direction_key)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (claim_id, chunk_id) DO NOTHING;
                    """,
                    (claim_id, chunk_id, direction_key),
                )

    def confirm(
        self,
        claim_id: int,
        citation_chunk_ids: list[int],
        direction_key: str,
        hint_confidence: float = 0.5,
    ) -> None:
        """Bayesian noisy-OR rollup: confidence ← 1 - (1 - old) * (1 - hint).

        Каждое подтверждение «приближает» уверенность к 1, причём вклад
        нового свидетельства зависит от его собственного confidence_hint.
        """
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET times_confirmed = times_confirmed + 1,
                    last_confirmed_at = now(),
                    confidence = LEAST(1.0,
                        1.0 - (1.0 - confidence) * (1.0 - GREATEST(0.0, LEAST(1.0, %s::real)))
                    )
                WHERE id = %s;
                """,
                (hint_confidence, claim_id),
            )
        self.add_citations(direction_key, claim_id, citation_chunk_ids)

    def set_canonical_predicate(self, claim_id: int, predicate_id: int) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET canonical_predicate_id = %s
                WHERE id = %s;
                """,
                (predicate_id, claim_id),
            )

    def set_tier(self, claim_id: int, tier: str) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET tier = %s::rag_v7.claim_tier
                WHERE id = %s;
                """,
                (tier, claim_id),
            )

    def supersede(
        self,
        direction_key: str,
        old_claim_id: int,
        new_claim_id: int,
        reason: str,
        decided_by: str = "llm_arbiter",
    ) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.claim_supersedes
                    (old_claim_id, new_claim_id, direction_key, reason, decided_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (old_claim_id, new_claim_id) DO NOTHING;
                """,
                (old_claim_id, new_claim_id, direction_key, reason, decided_by),
            )
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET status = 'superseded',
                    superseded_by_id = %s
                WHERE id = %s;
                """,
                (new_claim_id, old_claim_id),
            )

    def add_contradiction(
        self,
        direction_key: str,
        claim_a_id: int,
        claim_b_id: int,
        notes: str,
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_v7.claim_contradictions
                    (direction_key, claim_a_id, claim_b_id, notes)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (direction_key, claim_a_id, claim_b_id, notes),
            )
            contradiction_id = cur.fetchone()["id"]
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET status = 'flagged_contradiction'
                WHERE id IN (%s, %s) AND status = 'active';
                """,
                (claim_a_id, claim_b_id),
            )
            return contradiction_id

    def resolve_contradiction(
        self,
        contradiction_id: int,
        winner_id: int,
        notes: str | None = None,
    ) -> None:
        """Помечает противоречие resolved и снимает флаг с winner-а."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claim_contradictions
                SET status = 'resolved',
                    resolved_at = now(),
                    notes = COALESCE(%s, notes)
                WHERE id = %s;
                """,
                (notes, contradiction_id),
            )
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET status = 'active'
                WHERE id = %s AND status = 'flagged_contradiction';
                """,
                (winner_id,),
            )

    def clear_flag(self, claim_id: int) -> None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET status = 'active'
                WHERE id = %s AND status = 'flagged_contradiction';
                """,
                (claim_id,),
            )

    def claims_for_entity(
        self, direction_key: str, entity_id: int, only_active: bool = True
    ) -> list[dict[str, Any]]:
        status_clause = "AND status = 'active'" if only_active else ""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.predicate, c.canonical_predicate_id,
                       c.object_kind, c.object_entity_id, c.object_text,
                       c.claim_text, c.confidence, c.times_confirmed, c.tier,
                       c.first_seen_at, c.last_confirmed_at, c.status,
                       e.canonical_name AS object_canonical_name,
                       cp.canonical AS canonical_predicate
                FROM rag_v7.claims c
                LEFT JOIN rag_v7.entities e ON e.id = c.object_entity_id
                LEFT JOIN rag_v7.canonical_predicates cp ON cp.id = c.canonical_predicate_id
                WHERE c.direction_key = %s
                  AND c.subject_entity_id = %s
                  {status_clause}
                ORDER BY c.last_confirmed_at DESC, c.id;
                """,
                (direction_key, entity_id),
            )
            return cur.fetchall()

    def count_contradictions_for_entity(
        self, direction_key: str, entity_id: int
    ) -> int:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS n
                FROM rag_v7.claims
                WHERE direction_key = %s
                  AND subject_entity_id = %s
                  AND status = 'flagged_contradiction';
                """,
                (direction_key, entity_id),
            )
            return cur.fetchone()["n"]

    def documents_for_entity(
        self, direction_key: str, entity_id: int
    ) -> dict[int, int]:
        """Возвращает {document_id: claim_count} для provenance страницы."""
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT ch.document_id, count(DISTINCT c.id) AS claim_count
                FROM rag_v7.claims c
                JOIN rag_v7.claim_citations cc ON cc.claim_id = c.id
                JOIN rag_v7.chunks ch ON ch.id = cc.chunk_id
                WHERE c.direction_key = %s
                  AND c.subject_entity_id = %s
                  AND c.status = 'active'
                GROUP BY ch.document_id;
                """,
                (direction_key, entity_id),
            )
            return {row["document_id"]: row["claim_count"] for row in cur.fetchall()}

    def promote_tiers(
        self,
        direction_key: str,
        episodic_min_confirmations: int,
        semantic_min_confirmations: int,
        semantic_min_age_days: int,
    ) -> None:
        """Массовое обновление tier-ов по правилам promotion.

        - working → episodic: times_confirmed >= episodic_min_confirmations
        - episodic → semantic: times_confirmed >= semantic_min_confirmations
                               AND age >= semantic_min_age_days
        - procedural — пока не авто-promotion, резерв.
        """
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET tier = 'episodic'
                WHERE direction_key = %s
                  AND tier = 'working'
                  AND times_confirmed >= %s
                  AND status = 'active';
                """,
                (direction_key, episodic_min_confirmations),
            )
            cur.execute(
                """
                UPDATE rag_v7.claims
                SET tier = 'semantic'
                WHERE direction_key = %s
                  AND tier = 'episodic'
                  AND times_confirmed >= %s
                  AND first_seen_at <= now() - (%s || ' days')::interval
                  AND status = 'active';
                """,
                (
                    direction_key,
                    semantic_min_confirmations,
                    str(semantic_min_age_days),
                ),
            )
