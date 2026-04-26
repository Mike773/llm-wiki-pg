"""WikiQuery — самодостаточный модуль выборки для rag_v7_wiki.

Один публичный класс `WikiQuery` + всё, что ему нужно для работы
(connection-pool с pgvector, internal Pydantic-схемы для structured-LLM,
duck-typing-протоколы для Embedder/LLM). Зависит только от внешних
библиотек: psycopg, psycopg_pool, pgvector, pydantic. Подставляется в
чужой проект копированием одного файла.

Использование:

    from rag_v7_wiki.query import WikiQuery  # либо просто импорт этого файла

    with WikiQuery(
        connection_string="postgresql://user:pass@host/db",
        direction_key="research",
        llm=my_llm,
        embedder=my_embedder,
    ) as q:
        result = q.ask("Кто такой Alice?")
        print(result["answer"])
        print(result["report"])
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, TypeVar, runtime_checkable

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Duck-typed protocols — твои LLM и Embedder должны соответствовать этим
# интерфейсам (любой класс с такими методами автоматически подходит).
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...

    @property
    def model_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Inline Postgres helpers (без зависимости от rag_v7_wiki.dao.connection).
# ---------------------------------------------------------------------------


def _to_vec(values: list[float] | None) -> Vector | None:
    return None if values is None else Vector(values)


class _ConnectionManager:
    """Тонкая read-only обёртка над psycopg ConnectionPool с pgvector.

    Принимает либо DSN-строку (создаёт собственный pool, закрывает в close()),
    либо готовый `ConnectionPool` (тогда внешним владельцем управляем не мы).
    """

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

    def close(self) -> None:
        if self._owned:
            self._pool.close()


# ---------------------------------------------------------------------------
# Internal Pydantic schemas (LLM I/O для retrieval-стадий).
# ---------------------------------------------------------------------------


class _SlugPickResponse(BaseModel):
    """Ответ LLM в режиме wiki-only — какие slug-и из index-страницы релевантны."""

    relevant_slugs: list[str] = Field(default_factory=list)
    reasoning: str = ""


class _WikiAnswer(BaseModel):
    """Финальный ответ LLM, структурированный."""

    answer: str = Field(description="Markdown-ответ пользователю.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    insufficient_evidence: bool = Field(
        default=False,
        description="True, если в контексте недостаточно данных для уверенного ответа.",
    )
    cited_page_slugs: list[str] = Field(default_factory=list)
    cited_claim_ids: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# WikiQuery
# ---------------------------------------------------------------------------


_TIER_ORDER = ("working", "episodic", "semantic", "procedural")
_DEFAULT_PAGE_KINDS = ("entity", "source", "concept", "comparison", "overview", "index")


_ANSWER_SYSTEM = (
    "Ты отвечаешь на вопрос пользователя строго по предоставленному контексту "
    "из knowledge wiki. Правила:\n"
    "- Используй только факты из RELEVANT PAGES и RELEVANT CLAIMS.\n"
    "- Если контекста недостаточно — выставь insufficient_evidence=true и "
    "честно объясни, чего не хватает.\n"
    "- Цитируй страницы как [[slug]] и claim-ы как claim:N (числовой id).\n"
    "- Если видишь flagged_contradiction — упомяни оба варианта, не выбирай "
    "произвольно.\n"
    "- Будь сжат: один-два абзаца обычно достаточно."
)


_SLUG_PICK_SYSTEM = (
    "Тебе дан catalogue (index) wiki и вопрос пользователя. Верни список slug-ов "
    "страниц, которые могут содержать ответ. Если ничего не подходит — пустой "
    "список. Slug-и бери ровно из индекса, не выдумывай."
)


class WikiQuery:
    """Один-классовая реализация query/retrieval для rag_v7_wiki.

    Конфигурируется в __init__, вызывается через `ask(question)`. Возвращает
    `dict` с ключами `answer`, `report`, `elapsed_seconds`.
    """

    def __init__(
        self,
        *,
        connection_string: str | ConnectionPool,
        direction_key: str,
        llm: LLM,
        embedder: Embedder,
        # retrieval scope
        use_embeddings: bool = True,
        include_graph_expansion: bool = False,
        # top-K
        top_k_pages: int = 5,
        top_k_claims: int = 10,
        top_k_entities: int = 5,
        # filters
        min_similarity: float = 0.4,
        include_page_kinds: list[str] | None = None,
        include_superseded: bool = False,
        include_flagged_contradictions: bool = True,
        tier_floor: str = "working",
        # context budget
        max_context_chars: int = 12000,
        max_pages_in_context: int = 8,
        max_claims_in_context: int = 30,
        max_chars_per_page: int = 2500,
        # response
        require_citations: bool = True,
    ) -> None:
        if tier_floor not in _TIER_ORDER:
            raise ValueError(
                f"tier_floor must be one of {_TIER_ORDER}, got {tier_floor!r}"
            )
        self._cm = _ConnectionManager(connection_string)
        self.direction_key = direction_key
        self.llm = llm
        self.embedder = embedder

        self.use_embeddings = use_embeddings
        self.include_graph_expansion = include_graph_expansion
        self.top_k_pages = top_k_pages
        self.top_k_claims = top_k_claims
        self.top_k_entities = top_k_entities
        self.min_similarity = min_similarity
        self.include_page_kinds = (
            list(include_page_kinds) if include_page_kinds else list(_DEFAULT_PAGE_KINDS)
        )
        self.include_superseded = include_superseded
        self.include_flagged_contradictions = include_flagged_contradictions
        self.tier_floor = tier_floor
        self.max_context_chars = max_context_chars
        self.max_pages_in_context = max_pages_in_context
        self.max_claims_in_context = max_claims_in_context
        self.max_chars_per_page = max_chars_per_page
        self.require_citations = require_citations

    def __enter__(self) -> "WikiQuery":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._cm.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        report: dict[str, Any] = {
            "query": question,
            "direction_key": self.direction_key,
            "mode": "embeddings" if self.use_embeddings else "wiki_only",
            "settings": self._effective_settings(),
            "stages": [],
            "errors": [],
        }
        answer: str | None = None
        try:
            if self.use_embeddings:
                pages, claims, entities = self._retrieve_with_embeddings(question, report)
            else:
                pages, claims, entities = self._retrieve_wiki_only(question, report)

            context_text = self._compose_context(question, pages, claims, entities, report)

            if not context_text.strip() or (not pages and not claims and not entities):
                report["stages"].append(
                    {"name": "synthesize", "skipped": True, "reason": "empty_context"}
                )
                return {
                    "answer": (
                        "Недостаточно данных в вики, чтобы ответить на этот вопрос."
                    ),
                    "report": {
                        **report,
                        "answer_meta": {
                            "confidence": 0.0,
                            "insufficient_evidence": True,
                            "cited_page_slugs": [],
                            "cited_claim_ids": [],
                        },
                    },
                    "elapsed_seconds": time.perf_counter() - t0,
                }

            answer_obj = self._synthesize_answer(question, context_text, report)
            answer = answer_obj.answer
            report["answer_meta"] = {
                "confidence": answer_obj.confidence,
                "insufficient_evidence": answer_obj.insufficient_evidence,
                "cited_page_slugs": answer_obj.cited_page_slugs,
                "cited_claim_ids": answer_obj.cited_claim_ids,
            }
        except Exception as exc:
            report["errors"].append({"type": exc.__class__.__name__, "message": str(exc)})

        return {
            "answer": answer,
            "report": report,
            "elapsed_seconds": time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------
    # Retrieval — embeddings mode
    # ------------------------------------------------------------------

    def _retrieve_with_embeddings(
        self, question: str, report: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        [embedding] = self.embedder.embed([question])
        report["stages"].append({"name": "embed_query", "dim": len(embedding)})

        entities = self._sql_top_entities(embedding)
        report["stages"].append(
            {
                "name": "retrieve_entities",
                "count": len(entities),
                "results": [
                    {
                        "id": e["id"],
                        "name": e["canonical_name"],
                        "type": e["entity_type"],
                        "similarity": round(float(e["similarity"]), 4),
                    }
                    for e in entities
                ],
            }
        )

        claims = self._sql_top_claims(embedding)
        report["stages"].append(
            {
                "name": "retrieve_claims",
                "count": len(claims),
                "results": [
                    {
                        "id": c["id"],
                        "subject": c["subject_name"],
                        "predicate": c["predicate"],
                        "object": c["object_repr"],
                        "tier": c["tier"],
                        "status": c["status"],
                        "similarity": round(float(c["similarity"]), 4),
                    }
                    for c in claims
                ],
            }
        )

        pages = self._sql_top_pages(embedding)
        report["stages"].append(
            {
                "name": "retrieve_pages",
                "count": len(pages),
                "results": [
                    {
                        "id": p["id"],
                        "slug": p["slug"],
                        "title": p["title"],
                        "kind": p["page_kind"],
                        "similarity": round(float(p["similarity"]), 4),
                    }
                    for p in pages
                ],
            }
        )

        if self.include_graph_expansion and entities:
            seed_ids = [e["id"] for e in entities]
            neighbor_ids = self._sql_graph_neighbors(seed_ids)
            new_neighbors = [nid for nid in neighbor_ids if nid not in seed_ids]
            extra_pages = (
                self._sql_pages_for_entities(new_neighbors) if new_neighbors else []
            )
            existing_ids = {p["id"] for p in pages}
            extra_pages = [p for p in extra_pages if p["id"] not in existing_ids]
            pages.extend(extra_pages)
            report["stages"].append(
                {
                    "name": "graph_expansion",
                    "neighbor_entities": len(new_neighbors),
                    "added_pages": len(extra_pages),
                }
            )
        else:
            report["stages"].append({"name": "graph_expansion", "skipped": True})

        return pages, claims, entities

    # ------------------------------------------------------------------
    # Retrieval — wiki-only mode
    # ------------------------------------------------------------------

    def _retrieve_wiki_only(
        self, question: str, report: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        index_md = self._sql_get_singleton_content("index")
        if not index_md:
            report["stages"].append(
                {"name": "wiki_only_pick", "skipped": True, "reason": "no_index_page"}
            )
            return [], [], []

        user_prompt = (
            f"Вопрос пользователя:\n{question}\n\n"
            f"Index текущей вики:\n{index_md}"
        )
        pick = self.llm.structured(_SLUG_PICK_SYSTEM, user_prompt, _SlugPickResponse)
        slugs = [s.strip() for s in pick.relevant_slugs if s and s.strip()]

        report["stages"].append(
            {
                "name": "wiki_only_pick",
                "slugs": slugs,
                "reasoning": pick.reasoning[:500],
            }
        )

        if not slugs:
            return [], [], []

        pages = self._sql_pages_by_slug(slugs)
        report["stages"].append(
            {
                "name": "retrieve_pages",
                "count": len(pages),
                "results": [
                    {
                        "id": p["id"],
                        "slug": p["slug"],
                        "title": p["title"],
                        "kind": p["page_kind"],
                    }
                    for p in pages
                ],
            }
        )
        return pages, [], []

    # ------------------------------------------------------------------
    # Context composition
    # ------------------------------------------------------------------

    def _compose_context(
        self,
        question: str,
        pages: list[dict],
        claims: list[dict],
        entities: list[dict],
        report: dict[str, Any],
    ) -> str:
        budget = self.max_context_chars
        chunks: list[str] = []
        included_page_ids: list[int] = []
        included_claim_ids: list[int] = []

        chunks.append(f"QUESTION:\n{question}\n")

        if pages:
            chunks.append("RELEVANT PAGES:")
            for page in pages[: self.max_pages_in_context]:
                if budget <= 0:
                    break
                content = (page.get("content_md") or "").strip()
                if len(content) > self.max_chars_per_page:
                    content = content[: self.max_chars_per_page].rstrip() + "\n…(truncated)"
                sim = page.get("similarity")
                sim_str = f", sim={float(sim):.2f}" if sim is not None else ""
                header = (
                    f"\n### [[{page['title']}]] (slug={page['slug']}, "
                    f"kind={page['page_kind']}{sim_str})"
                )
                block = header + "\n" + content
                if len(block) > budget:
                    block = block[:budget].rstrip() + "\n…(budget cut)"
                chunks.append(block)
                budget -= len(block)
                included_page_ids.append(page["id"])

        if claims and budget > 0:
            chunks.append("\nRELEVANT CLAIMS:")
            for claim in claims[: self.max_claims_in_context]:
                if budget <= 0:
                    break
                tier = claim.get("tier", "working")
                conf = float(claim.get("confidence", 0))
                times = int(claim.get("times_confirmed", 1))
                status = claim.get("status", "active")
                line = (
                    f"- claim:{claim['id']}  {claim['subject_name']} → "
                    f"{claim['predicate']} → {claim['object_repr']} "
                    f"(×{times}, conf={conf:.2f}, {tier})"
                )
                if status == "flagged_contradiction":
                    line += "  ⚠ flagged_contradiction"
                if len(line) > budget:
                    break
                chunks.append(line)
                budget -= len(line)
                included_claim_ids.append(claim["id"])

        if entities and budget > 0:
            chunks.append("\nKNOWN ENTITIES:")
            for ent in entities:
                if budget <= 0:
                    break
                line = (
                    f"- {ent['canonical_name']} ({ent['entity_type']}, "
                    f"mentions={ent.get('mention_count', 0)})"
                )
                if len(line) > budget:
                    break
                chunks.append(line)
                budget -= len(line)

        text = "\n".join(chunks)
        report["stages"].append(
            {
                "name": "compose_context",
                "context_chars": len(text),
                "page_ids": included_page_ids,
                "claim_ids": included_claim_ids,
                "budget_remaining": max(0, budget),
            }
        )
        return text

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize_answer(
        self, question: str, context: str, report: dict[str, Any]
    ) -> _WikiAnswer:
        instructions: list[str] = []
        if self.require_citations:
            instructions.append(
                "Цитируй страницы как [[slug]] и claim-ы как claim:N — без них ответ "
                "не считается полным."
            )

        user_prompt = context
        if instructions:
            user_prompt += "\n\nINSTRUCTIONS:\n" + "\n".join(f"- {i}" for i in instructions)

        answer_obj = self.llm.structured(_ANSWER_SYSTEM, user_prompt, _WikiAnswer)
        report["stages"].append(
            {
                "name": "synthesize",
                "model": getattr(self.llm, "model_name", "unknown"),
                "prompt_chars": len(user_prompt),
                "answer_chars": len(answer_obj.answer),
            }
        )
        return answer_obj

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    def _sql_top_entities(self, embedding: list[float]) -> list[dict[str, Any]]:
        q = _to_vec(embedding)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity_type, canonical_name, salient_attrs,
                       mention_count, confidence,
                       1 - (canonical_name_embedding <=> %s) AS similarity
                FROM rag_v7.entities
                WHERE direction_key = %s
                ORDER BY canonical_name_embedding <=> %s
                LIMIT %s;
                """,
                (q, self.direction_key, q, self.top_k_entities),
            )
            rows = cur.fetchall()
        return [r for r in rows if (r.get("similarity") or 0.0) >= self.min_similarity]

    def _sql_top_claims(self, embedding: list[float]) -> list[dict[str, Any]]:
        q = _to_vec(embedding)
        statuses: list[str] = ["active"]
        if self.include_flagged_contradictions:
            statuses.append("flagged_contradiction")
        if self.include_superseded:
            statuses.append("superseded")

        floor_idx = _TIER_ORDER.index(self.tier_floor)
        tiers = list(_TIER_ORDER[floor_idx:])

        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.predicate, c.claim_text, c.confidence,
                       c.times_confirmed, c.tier::text AS tier, c.status::text AS status,
                       c.last_confirmed_at,
                       e1.canonical_name AS subject_name,
                       COALESCE(e2.canonical_name, c.object_text) AS object_repr,
                       1 - (c.claim_embedding <=> %s) AS similarity
                FROM rag_v7.claims c
                JOIN rag_v7.entities e1 ON e1.id = c.subject_entity_id
                LEFT JOIN rag_v7.entities e2 ON e2.id = c.object_entity_id
                WHERE c.direction_key = %s
                  AND c.status::text = ANY(%s)
                  AND c.tier::text = ANY(%s)
                ORDER BY c.claim_embedding <=> %s
                LIMIT %s;
                """,
                (q, self.direction_key, statuses, tiers, q, self.top_k_claims),
            )
            rows = cur.fetchall()
        return [r for r in rows if (r.get("similarity") or 0.0) >= self.min_similarity]

    def _sql_top_pages(self, embedding: list[float]) -> list[dict[str, Any]]:
        q = _to_vec(embedding)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_kind::text AS page_kind, slug, title, content_md,
                       quality_score, coverage_claims, coverage_contradictions,
                       1 - (content_embedding <=> %s) AS similarity
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s
                  AND page_kind::text = ANY(%s)
                ORDER BY content_embedding <=> %s
                LIMIT %s;
                """,
                (q, self.direction_key, self.include_page_kinds, q, self.top_k_pages),
            )
            rows = cur.fetchall()
        return [r for r in rows if (r.get("similarity") or 0.0) >= self.min_similarity]

    def _sql_get_singleton_content(self, page_kind: str) -> str | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT content_md FROM rag_v7.wiki_pages
                WHERE direction_key = %s
                  AND page_kind = %s::rag_v7.wiki_page_kind;
                """,
                (self.direction_key, page_kind),
            )
            row = cur.fetchone()
        return row["content_md"] if row else None

    def _sql_pages_by_slug(self, slugs: list[str]) -> list[dict[str, Any]]:
        if not slugs:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_kind::text AS page_kind, slug, title, content_md,
                       quality_score, coverage_claims, coverage_contradictions
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s AND slug = ANY(%s);
                """,
                (self.direction_key, slugs),
            )
            return cur.fetchall()

    def _sql_graph_neighbors(self, seed_ids: list[int]) -> list[int]:
        if not seed_ids:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH seed AS (SELECT unnest(%s::bigint[]) AS id)
                SELECT DISTINCT
                    CASE WHEN c.subject_entity_id = s.id
                         THEN c.object_entity_id ELSE c.subject_entity_id END AS neighbor_id
                FROM rag_v7.claims c
                JOIN seed s ON s.id IN (c.subject_entity_id, c.object_entity_id)
                WHERE c.direction_key = %s AND c.status = 'active'
                  AND c.object_entity_id IS NOT NULL;
                """,
                (seed_ids, self.direction_key),
            )
            rows = cur.fetchall()
        return [r["neighbor_id"] for r in rows if r["neighbor_id"] is not None]

    def _sql_pages_for_entities(self, entity_ids: list[int]) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_kind::text AS page_kind, slug, title, content_md,
                       quality_score, coverage_claims, coverage_contradictions,
                       NULL::real AS similarity
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s AND entity_id = ANY(%s)
                  AND page_kind::text = ANY(%s);
                """,
                (self.direction_key, entity_ids, self.include_page_kinds),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _effective_settings(self) -> dict[str, Any]:
        return {
            "use_embeddings": self.use_embeddings,
            "include_graph_expansion": self.include_graph_expansion,
            "top_k_pages": self.top_k_pages,
            "top_k_claims": self.top_k_claims,
            "top_k_entities": self.top_k_entities,
            "min_similarity": self.min_similarity,
            "include_page_kinds": self.include_page_kinds,
            "include_superseded": self.include_superseded,
            "include_flagged_contradictions": self.include_flagged_contradictions,
            "tier_floor": self.tier_floor,
            "max_context_chars": self.max_context_chars,
            "max_pages_in_context": self.max_pages_in_context,
            "max_claims_in_context": self.max_claims_in_context,
            "max_chars_per_page": self.max_chars_per_page,
            "require_citations": self.require_citations,
        }
