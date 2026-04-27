"""KnowledgeExtractor — самодостаточный модуль для агента-аналитика.

Из базы знаний (rag_v7 schema, PostgreSQL+pgvector) собирает business-контекст,
связанный с конкретной инструкцией, относительно найденной должности/роли.
LLM **НЕ отвечает** на инструкцию — только формирует подсказку из фактов.

Модуль самодостаточен: один файл, никаких импортов из rag_v7_wiki. Зависимости —
только внешние библиотеки: psycopg, psycopg_pool, pgvector, pydantic. Можно
скопировать во внешний проект и пользоваться, передав свои llm/embedder/DSN.

Использование:

    from knowledge_extractor import KnowledgeExtractor

    with KnowledgeExtractor(my_llm, my_embedder, "postgresql://...") as kx:
        result = kx.extract(
            direction_key="research",
            position_or_role="руководитель отдела продаж",
            instruction="Какие KPI у этой роли в Q3?",
        )
        print(result["answer"])
        print(result["trace"])

Pipeline:
    1) Найти наиболее релевантную сущность (entity) под position_or_role:
       semantic top-K по rag_v7.entities.canonical_name_embedding (БЕЗ фильтра по
       entity_type) → LLM-арбитр выбирает одну.
    2) Multi-source retrieval с biased на найденную сущность: entity wiki page,
       entity-biased claims, general claims similar to instruction, similar
       pages, source chunks, 2-hop typed graph traversal по predicate.
    3) Multi-signal Python re-rank: cosine + confidence + tier + recency
       (для claims) и cosine + quality_score + source-bonus (для pages).
    4) LLM-арбитр для всех flagged_contradiction пар (оба варианта остаются в
       контексте, арбитр даёт лишь подсказку).
    5) LLM-синтез: extracts business-контекст; **запрещено** отвечать на
       instruction.

Возвращает dict с двумя текстовыми полями:
    - "answer": синтезированный business-контекст (markdown);
    - "trace":  numbered narrative «как получили и откуда» (markdown).

Вся выборка фильтруется по direction_key. Embeddings ожидаются 2560-мерные
(жёсткое требование схемы rag_v7).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
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
    """Read-only обёртка над psycopg ConnectionPool с регистрацией pgvector.

    Принимает либо DSN-строку (создаёт собственный pool, закрывает в close()),
    либо готовый ConnectionPool (внешним владельцем не управляем).
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
# Constants
# ---------------------------------------------------------------------------


_EXPECTED_EMBEDDING_DIM = 2560
_DEFAULT_PAGE_KINDS = ("entity", "source", "concept", "comparison", "overview", "index")
_TIER_SCORE_MAP = {"working": 0.0, "episodic": 0.4, "semantic": 0.7, "procedural": 1.0}
_ACTIVE_STATUSES = ("active", "flagged_contradiction")


# ---------------------------------------------------------------------------
# Internal Pydantic schemas
# ---------------------------------------------------------------------------


class _EntityArbiterResponse(BaseModel):
    """Решение LLM-арбитра — какая сущность из top-K кандидатов наиболее
    соответствует запросу про должность/роль."""

    matched_entity_id: int | None = Field(
        default=None,
        description="ID выбранной сущности из списка кандидатов или null, если ни одна не подходит.",
    )
    reasoning: str = Field(default="", description="1-3 предложения объяснения выбора.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _ContradictionArbiterResponse(BaseModel):
    """Решение LLM-арбитра — какой из двух противоречащих claim'ов более вероятен."""

    likely_correct_claim_id: int | None = Field(
        default=None,
        description="claim_id того, что более вероятен; null — если данных недостаточно.",
    )
    reasoning: str = Field(
        default="",
        description="1-3 предложения про recency/authority/число подтверждений/цитаты.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _KnowledgeContextResponse(BaseModel):
    """Финальный синтез: business-контекст для instruction. **НЕ ответ на неё.**"""

    context_summary: str = Field(
        description="Связанный business-контекст из базы знаний, релевантный для instruction. "
        "НЕ ответ на instruction — а факты/детали, которые помогут на него ответить."
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Атомарные факты-цитаты с указанием источника ([[slug]] / claim:N / chunk:N).",
    )
    coverage_notes: str = Field(
        default="",
        description="Что в базе знаний есть и чего не хватает для instruction (явно).",
    )
    cited_page_slugs: list[str] = Field(default_factory=list)
    cited_claim_ids: list[int] = Field(default_factory=list)
    cited_chunk_ids: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompts (RU)
# ---------------------------------------------------------------------------


_ENTITY_ARBITER_SYSTEM = (
    "Тебе дан запрос пользователя — описание позиции, роли или человека — "
    "и список кандидатов из базы сущностей (canonical_name, entity_type, "
    "salient_attrs, similarity). Выбери ОДНУ сущность, наиболее релевантную "
    "запросу. Правила:\n"
    "- Решающий критерий — смысловое соответствие, а не только косинусная "
    "близость имени.\n"
    "- Учитывай entity_type: позиция, роль, человек, отдел, проект — это "
    "РАЗНЫЕ типы. Если запрос «руководитель отдела X», а кандидат типа "
    "person — это ок только если salient_attrs подтверждают связь с ролью.\n"
    "- Если ни один кандидат не подходит — верни matched_entity_id=null и "
    "объясни почему в reasoning.\n"
    "- Не выдумывай ID, бери только из списка."
)


_CONTRADICTION_ARBITER_SYSTEM = (
    "Дано два противоречивых claim'а с одной парой (subject, predicate). "
    "Выбери более вероятный, опираясь на:\n"
    "(а) last_confirmed_at (новее ⇒ выше),\n"
    "(б) times_confirmed (больше ⇒ выше),\n"
    "(в) confidence,\n"
    "(г) дословные цитаты из source chunks.\n"
    "Если данных недостаточно для решения — likely_correct_claim_id=null. "
    "Это лишь подсказка для синтезатора; оба claim'а останутся в его "
    "контексте. Не выдумывай ID, бери только из тех двух, что показаны."
)


_KNOWLEDGE_SYNTHESIS_SYSTEM = (
    "Ты НЕ отвечаешь на instruction. Ты формируешь BUSINESS-КОНТЕКСТ из базы "
    "знаний, который поможет другому агенту ответить на instruction.\n\n"
    "Правила:\n"
    "- ЗАПРЕЩЕНО давать прямой ответ, рекомендации, выводы, советы по "
    "instruction. Запрещено формулировать «ответ» — только факты и контекст.\n"
    "- Используй ТОЛЬКО факты из предоставленного контекста (PRIMARY ENTITY, "
    "RELEVANT PAGES, RELEVANT CLAIMS, GRAPH-DERIVED CLAIMS, SOURCE EXCERPTS, "
    "CONTRADICTION HINTS).\n"
    "- Цитируй ДОСЛОВНО: имена, цифры, даты, формулировки. Никаких пересказов "
    "в «общих чертах». Если в источнике «23.4% в Q3 2025» — пиши именно так.\n"
    "- Каждый ключевой факт сопровождай ссылкой: страница [[slug]], claim "
    "claim:N, чанк chunk:N.\n"
    "- Если SOURCE EXCERPTS противоречат пересказу в RELEVANT PAGES — "
    "приоритет у SOURCE.\n"
    "- Для flagged_contradiction приводи ОБА варианта. Если есть CONTRADICTION "
    "HINTS — упомяни подсказку контр-арбитра, но НЕ как истину.\n"
    "- В coverage_notes явно пиши: какие аспекты instruction покрыты данными "
    "из базы знаний, какие — НЕТ.\n"
    "- Если данных мало — пусть context_summary будет коротким; не растягивай "
    "повторами и общими словами."
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _tier_score(tier: str | None) -> float:
    return _TIER_SCORE_MAP.get(tier or "", 0.0)


def _recency_score(
    last_confirmed_at: datetime | None,
    now: datetime,
    half_life_days: float,
) -> float:
    if last_confirmed_at is None or half_life_days <= 0:
        return 0.0
    age_days = (now - last_confirmed_at).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    return 0.5 ** (age_days / half_life_days)


def _claim_score(
    row: dict[str, Any],
    *,
    w_sim: float,
    w_conf: float,
    w_tier: float,
    w_recency: float,
    half_life_days: float,
    now: datetime,
) -> dict[str, float]:
    """Возвращает {final, sim, conf, tier, recency} — компоненты для трассы."""
    sim = float(row.get("similarity") or 0.0)
    conf = float(row.get("confidence") or 0.0)
    tier_s = _tier_score(row.get("tier"))
    rec_s = _recency_score(row.get("last_confirmed_at"), now, half_life_days)
    final = w_sim * sim + w_conf * conf + w_tier * tier_s + w_recency * rec_s
    return {
        "final": final,
        "sim": sim,
        "conf": conf,
        "tier_s": tier_s,
        "recency_s": rec_s,
    }


def _page_score(
    row: dict[str, Any],
    *,
    w_sim: float,
    w_quality: float,
    w_source_bonus: float,
) -> dict[str, float]:
    sim = float(row.get("similarity") or 0.0)
    quality = float(row.get("quality_score") or 0.0)
    is_source = 1.0 if row.get("page_kind") == "source" else 0.0
    final = w_sim * sim + w_quality * quality + w_source_bonus * is_source
    return {
        "final": final,
        "sim": sim,
        "quality": quality,
        "is_source": is_source,
    }


# ---------------------------------------------------------------------------
# KnowledgeExtractor
# ---------------------------------------------------------------------------


class KnowledgeExtractor:
    """Standalone knowledge-extraction для analyst-агента.

    Из (direction_key, position_or_role, instruction) формирует:
      - синтез релевантного business-контекста (НЕ ответ на instruction);
      - текстовый трассирующий отчёт о пайплайне и источниках.

    Конфигурируется тремя зависимостями (llm, embedder, db_connection_string),
    остальное — разумные дефолты. Контекст-менеджер для закрытия pool-а.
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        db_connection_string: str | ConnectionPool,
        *,
        # entity-resolution
        entity_top_k: int = 8,
        entity_min_similarity: float = 0.25,
        # retrieval — финальные top-K (после re-rank)
        top_k_entity_biased_claims: int = 12,
        top_k_general_claims: int = 8,
        top_k_pages: int = 6,
        top_k_chunks_per_entity: int = 4,
        top_k_chunks_per_claim: int = 6,
        # pool до re-rank (берём pool_multiplier × top_k из SQL)
        pool_multiplier: int = 3,
        # graph
        graph_enabled: bool = True,
        graph_max_hops: int = 2,
        graph_hop1_limit: int = 8,
        graph_hop2_limit: int = 12,
        graph_min_confidence: float = 0.4,
        # contradictions
        contradiction_arbiter_enabled: bool = True,
        contradiction_arbiter_max_pairs: int = 4,
        # ranking weights — claims
        claim_w_sim: float = 0.6,
        claim_w_confidence: float = 0.2,
        claim_w_tier: float = 0.1,
        claim_w_recency: float = 0.1,
        recency_half_life_days: float = 30.0,
        # ranking weights — pages
        page_w_sim: float = 0.6,
        page_w_quality: float = 0.3,
        page_w_source_bonus: float = 0.1,
        # filters
        include_page_kinds: list[str] | None = None,
        # context budget
        max_context_chars: int = 24000,
        max_chars_per_page: int = 5000,
        max_chars_per_chunk: int = 1800,
    ) -> None:
        # Валидации
        if not 0.0 <= entity_min_similarity <= 1.0:
            raise ValueError(
                f"entity_min_similarity must be in [0,1], got {entity_min_similarity}"
            )
        for name, val in [
            ("entity_top_k", entity_top_k),
            ("top_k_entity_biased_claims", top_k_entity_biased_claims),
            ("top_k_general_claims", top_k_general_claims),
            ("top_k_pages", top_k_pages),
            ("top_k_chunks_per_entity", top_k_chunks_per_entity),
            ("top_k_chunks_per_claim", top_k_chunks_per_claim),
            ("pool_multiplier", pool_multiplier),
            ("graph_max_hops", graph_max_hops),
            ("graph_hop1_limit", graph_hop1_limit),
            ("graph_hop2_limit", graph_hop2_limit),
            ("contradiction_arbiter_max_pairs", contradiction_arbiter_max_pairs),
            ("max_context_chars", max_context_chars),
            ("max_chars_per_page", max_chars_per_page),
            ("max_chars_per_chunk", max_chars_per_chunk),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if not 0.0 <= graph_min_confidence <= 1.0:
            raise ValueError(
                f"graph_min_confidence must be in [0,1], got {graph_min_confidence}"
            )
        if recency_half_life_days <= 0:
            raise ValueError(
                f"recency_half_life_days must be > 0, got {recency_half_life_days}"
            )

        # Embedder dim hard requirement (схема pgvector vector(2560))
        try:
            actual_dim = embedder.dim
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "embedder.dim must be readable; got exception: " + repr(exc)
            ) from exc
        if actual_dim != _EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"This module targets vector({_EXPECTED_EMBEDDING_DIM}) columns; "
                f"embedder.dim={actual_dim}. Either re-embed your DB at "
                f"{_EXPECTED_EMBEDDING_DIM} dims or change vector column types."
            )

        self.llm = llm
        self.embedder = embedder
        self._cm = _ConnectionManager(db_connection_string)

        self.entity_top_k = entity_top_k
        self.entity_min_similarity = entity_min_similarity
        self.top_k_entity_biased_claims = top_k_entity_biased_claims
        self.top_k_general_claims = top_k_general_claims
        self.top_k_pages = top_k_pages
        self.top_k_chunks_per_entity = top_k_chunks_per_entity
        self.top_k_chunks_per_claim = top_k_chunks_per_claim
        self.pool_multiplier = pool_multiplier

        self.graph_enabled = graph_enabled
        self.graph_max_hops = graph_max_hops
        self.graph_hop1_limit = graph_hop1_limit
        self.graph_hop2_limit = graph_hop2_limit
        self.graph_min_confidence = graph_min_confidence

        self.contradiction_arbiter_enabled = contradiction_arbiter_enabled
        self.contradiction_arbiter_max_pairs = contradiction_arbiter_max_pairs

        self.claim_w_sim = claim_w_sim
        self.claim_w_confidence = claim_w_confidence
        self.claim_w_tier = claim_w_tier
        self.claim_w_recency = claim_w_recency
        self.recency_half_life_days = recency_half_life_days
        self.page_w_sim = page_w_sim
        self.page_w_quality = page_w_quality
        self.page_w_source_bonus = page_w_source_bonus

        self.include_page_kinds = (
            list(include_page_kinds) if include_page_kinds else list(_DEFAULT_PAGE_KINDS)
        )
        self.max_context_chars = max_context_chars
        self.max_chars_per_page = max_chars_per_page
        self.max_chars_per_chunk = max_chars_per_chunk

    def __enter__(self) -> "KnowledgeExtractor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._cm.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        direction_key: str,
        position_or_role: str,
        instruction: str,
    ) -> dict[str, str]:
        """Главный метод — собирает контекст и формирует ответ.

        Возвращает {'answer': <markdown business-контекст>, 'trace': <markdown narrative>}.
        Никогда не выкидывает наружу исключения (errors попадают в trace).
        """
        t0 = time.perf_counter()
        trace: list[str] = []

        # --- 0. Inputs
        trace.append(
            "## 0. Inputs\n"
            f"- direction_key: `{direction_key}`\n"
            f"- position_or_role: `{position_or_role!r}`\n"
            f"- instruction: `{instruction!r}`"
        )

        try:
            # --- 1. Direction sanity check
            if not self._sql_check_direction_exists(direction_key):
                trace.append(
                    "## 1. Direction check\n- exists: **false** — early exit"
                )
                return {
                    "answer": "",
                    "trace": self._format_trace(trace, time.perf_counter() - t0),
                }
            trace.append("## 1. Direction check\n- exists: true")

            # --- 2. Embed position_or_role + 3. Top-K candidates + 4. LLM arbitration
            matched_entity, role_emb_dim = self._resolve_entity(
                direction_key, position_or_role, trace
            )

            # Если нет кандидатов вообще — продолжаем пайплайн? Нет, без entity нет
            # entity-bias retrieval'а. Без instruction-similarity (см. ниже) тоже
            # ничего нет → выходим. Но если instr_emb получится — общий retrieval
            # ещё имеет смысл. Логика та же ниже: matched_entity=None допустимо.

            # --- 5. Embed instruction
            instr_emb: list[float] | None = None
            try:
                [instr_emb] = self.embedder.embed([instruction])
                trace.append(
                    "## 5. Embed instruction\n"
                    f"- dim: {len(instr_emb)}"
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    "## 5. Embed instruction\n"
                    f"- **error**: {exc.__class__.__name__}: {exc}\n"
                    "- продолжаем без instruction-similarity (entity-only)"
                )

            # --- 6. Pool retrieval
            pool = self._retrieve_pool(direction_key, matched_entity, instr_emb, trace)

            # --- 7. Graph traversal
            graph_claims = self._graph_traverse(
                direction_key, matched_entity, instr_emb, trace
            )
            # graph claims идут в общий пул claims перед re-rank
            pool["claims_general"].extend(
                c for c in graph_claims
                if c["id"] not in {x["id"] for x in pool["claims_entity"]}
            )

            # --- 8. Re-rank
            now = datetime.now(timezone.utc)
            ranked = self._rerank(pool, now, trace)

            # --- 9. Dedup
            final_pages, final_claims, final_chunks = self._dedup(
                direction_key=direction_key,
                pool=pool,
                ranked=ranked,
                trace=trace,
            )

            # --- 10. Contradiction arbiter
            contradiction_hints: list[dict[str, Any]] = []
            if self.contradiction_arbiter_enabled:
                contradiction_hints = self._arbitrate_contradictions(
                    direction_key, final_claims, trace
                )
            else:
                trace.append("## 9. Contradiction arbiter\n- disabled (config)")

            # --- 11. Compose context
            context_text, included = self._compose_context(
                position_or_role=position_or_role,
                instruction=instruction,
                matched_entity=matched_entity,
                pages=final_pages,
                claims=final_claims,
                chunks=final_chunks,
                contradiction_hints=contradiction_hints,
                trace=trace,
            )

            # --- 12. Empty-context guard
            if not context_text.strip() or (
                not final_pages and not final_claims and not final_chunks
            ):
                trace.append(
                    "## 11. Synthesis\n- **skipped**: empty context (база знаний "
                    "не содержит релевантного контекста для этого запроса)"
                )
                return {
                    "answer": "",
                    "trace": self._format_trace(trace, time.perf_counter() - t0),
                }

            # --- 13. Synthesis
            try:
                synth = self._synthesize(
                    position_or_role=position_or_role,
                    instruction=instruction,
                    context_text=context_text,
                    trace=trace,
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    "## 11. Synthesis\n"
                    f"- **error**: {exc.__class__.__name__}: {exc}"
                )
                return {
                    "answer": "",
                    "trace": self._format_trace(trace, time.perf_counter() - t0),
                }

            # --- 14. Format answer
            answer_text = self._format_answer(synth)

            # --- 15. Sources used
            trace.append(self._format_sources_used(synth, included))

            return {
                "answer": answer_text,
                "trace": self._format_trace(trace, time.perf_counter() - t0),
            }

        except Exception as exc:  # noqa: BLE001 — last-resort safety net
            trace.append(
                "## ! Unhandled exception\n"
                f"- {exc.__class__.__name__}: {exc}"
            )
            return {
                "answer": "",
                "trace": self._format_trace(trace, time.perf_counter() - t0),
            }

    # ------------------------------------------------------------------
    # Step 2-4: position/role resolution
    # ------------------------------------------------------------------

    def _resolve_entity(
        self,
        direction_key: str,
        position_or_role: str,
        trace: list[str],
    ) -> tuple[dict[str, Any] | None, int]:
        """Embed → top-K candidates → LLM arbiter → re-fetch full row."""

        # Step 2: embed
        try:
            [role_emb] = self.embedder.embed([position_or_role])
        except Exception as exc:  # noqa: BLE001
            trace.append(
                "## 2. Embed position_or_role\n"
                f"- **error**: {exc.__class__.__name__}: {exc}"
            )
            return None, 0
        trace.append(
            "## 2. Embed position_or_role\n"
            f"- dim: {len(role_emb)}"
        )

        # Step 3: top-K candidates
        candidates = self._sql_top_entity_candidates(
            direction_key, role_emb, self.entity_top_k
        )
        candidates = [
            r for r in candidates
            if (r.get("similarity") or 0.0) >= self.entity_min_similarity
        ]
        if not candidates:
            trace.append(
                f"## 3. Top-K candidates (k={self.entity_top_k}, "
                f"threshold={self.entity_min_similarity})\n"
                "- **no candidates** above threshold"
            )
            trace.append(
                "## 4. LLM arbitration\n- skipped: no candidates"
            )
            return None, len(role_emb)

        cand_lines = [
            f"- id={c['id']}  {c['canonical_name']!r} ({c['entity_type']}) "
            f"sim={float(c['similarity']):.3f}  attrs={self._compact_json(c.get('salient_attrs'))}"
            for c in candidates
        ]
        trace.append(
            f"## 3. Top-K candidates (k={self.entity_top_k}, "
            f"threshold={self.entity_min_similarity})\n"
            + "\n".join(cand_lines)
        )

        # Step 4: LLM arbitration
        arbiter_prompt = self._build_arbiter_prompt(position_or_role, candidates)
        candidate_ids = {int(c["id"]) for c in candidates}
        model_name = getattr(self.llm, "model_name", "unknown")
        try:
            decision = self.llm.structured(
                _ENTITY_ARBITER_SYSTEM, arbiter_prompt, _EntityArbiterResponse
            )
        except Exception as exc:  # noqa: BLE001
            trace.append(
                "## 4. LLM arbitration\n"
                f"- model: {model_name}\n"
                f"- **error**: {exc.__class__.__name__}: {exc}\n"
                "- продолжаем без entity-bias"
            )
            return None, len(role_emb)

        picked = decision.matched_entity_id
        if picked is not None and picked not in candidate_ids:
            trace.append(
                "## 4. LLM arbitration\n"
                f"- model: {model_name}\n"
                f"- LLM picked id={picked}, **NOT in candidate set**, treating as None\n"
                f"- reasoning: {decision.reasoning[:300]}"
            )
            return None, len(role_emb)

        if picked is None:
            trace.append(
                "## 4. LLM arbitration\n"
                f"- model: {model_name}\n"
                "- matched_entity_id: **null** (ни один кандидат не подходит)\n"
                f"- confidence: {decision.confidence:.2f}\n"
                f"- reasoning: {decision.reasoning[:300]}"
            )
            return None, len(role_emb)

        full_row = self._sql_fetch_entity_by_id(direction_key, int(picked))
        if not full_row:
            trace.append(
                "## 4. LLM arbitration\n"
                f"- model: {model_name}\n"
                f"- picked id={picked} not found by re-fetch (race?), treating as None"
            )
            return None, len(role_emb)

        trace.append(
            "## 4. LLM arbitration\n"
            f"- model: {model_name}\n"
            f"- matched_entity_id: **{picked}**  ({full_row['canonical_name']!r} / {full_row['entity_type']})\n"
            f"- confidence: {decision.confidence:.2f}\n"
            f"- reasoning: {decision.reasoning[:300]}"
        )
        return full_row, len(role_emb)

    def _build_arbiter_prompt(
        self, position_or_role: str, candidates: list[dict[str, Any]]
    ) -> str:
        lines = [
            "Запрос пользователя (позиция/роль/человек):",
            f"  {position_or_role!r}",
            "",
            "Кандидаты из базы сущностей:",
        ]
        for c in candidates:
            lines.append(
                f"  - id={c['id']}  canonical_name={c['canonical_name']!r}  "
                f"entity_type={c['entity_type']!r}  "
                f"similarity={float(c['similarity']):.3f}  "
                f"mention_count={c.get('mention_count', 0)}  "
                f"salient_attrs={self._compact_json(c.get('salient_attrs'))}"
            )
        lines.append("")
        lines.append(
            "Выбери ОДНУ сущность (matched_entity_id) или null, если ни одна не подходит."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 6: Pool retrieval
    # ------------------------------------------------------------------

    def _retrieve_pool(
        self,
        direction_key: str,
        matched_entity: dict[str, Any] | None,
        instr_emb: list[float] | None,
        trace: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Возвращает pool: entity_page (1 строка), claims_entity, claims_general,
        pages_general, chunks_entity. Каждый — список строк из БД (с similarity)."""

        pool: dict[str, list[dict[str, Any]]] = {
            "entity_page": [],
            "claims_entity": [],
            "claims_general": [],
            "pages_general": [],
            "chunks_entity": [],
        }
        sub_trace: list[str] = []

        # entity wiki page
        if matched_entity is not None:
            page = self._sql_wiki_page_for_entity(direction_key, int(matched_entity["id"]))
            if page:
                pool["entity_page"].append(page)
                sub_trace.append(
                    f"- entity wiki page: slug=`{page['slug']}` "
                    f"(quality={page.get('quality_score')})"
                )
            else:
                sub_trace.append("- entity wiki page: не найдена")
        else:
            sub_trace.append("- entity wiki page: skipped (no matched entity)")

        # entity-biased claims pool
        if matched_entity is not None and instr_emb is not None:
            pool_size = self.top_k_entity_biased_claims * self.pool_multiplier
            rows = self._sql_top_claims_about_entity(
                direction_key, int(matched_entity["id"]), instr_emb, pool_size
            )
            pool["claims_entity"] = rows
            sample_ids = [r["id"] for r in rows[:5]]
            sub_trace.append(
                f"- entity-biased claims pool: {len(rows)} (sample: {sample_ids})"
            )
        elif matched_entity is not None and instr_emb is None:
            # instruction embed упало — берём все entity-биас claims без ranking
            # by sim, отсортируем по confidence потом. Используем нулевой вектор
            # — но проще: пропускаем entity-biased если нет instr_emb. Trace.
            sub_trace.append(
                "- entity-biased claims pool: skipped (no instruction embedding)"
            )
        else:
            sub_trace.append(
                "- entity-biased claims pool: skipped (no matched entity)"
            )

        # general claims pool
        if instr_emb is not None:
            pool_size = self.top_k_general_claims * self.pool_multiplier
            rows = self._sql_top_claims_general(direction_key, instr_emb, pool_size)
            pool["claims_general"] = rows
            sample_ids = [r["id"] for r in rows[:5]]
            sub_trace.append(
                f"- general claims pool: {len(rows)} (sample: {sample_ids})"
            )
        else:
            sub_trace.append("- general claims pool: skipped (no instruction embedding)")

        # relevant pages pool
        if instr_emb is not None:
            pool_size = self.top_k_pages * self.pool_multiplier
            rows = self._sql_top_pages_general(direction_key, instr_emb, pool_size)
            pool["pages_general"] = rows
            sample_slugs = [r["slug"] for r in rows[:5]]
            sub_trace.append(
                f"- relevant pages pool: {len(rows)} (sample: {sample_slugs})"
            )
        else:
            sub_trace.append("- relevant pages pool: skipped (no instruction embedding)")

        # entity chunks via mentions
        if matched_entity is not None:
            rows = self._sql_chunks_for_entity(
                direction_key, int(matched_entity["id"]), self.top_k_chunks_per_entity
            )
            pool["chunks_entity"] = rows
            sample_ids = [r["chunk_id"] for r in rows[:5]]
            sub_trace.append(
                f"- entity chunks (mentions): {len(rows)} (sample: {sample_ids})"
            )
        else:
            sub_trace.append("- entity chunks: skipped (no matched entity)")

        trace.append("## 6. Pool retrieval (до re-rank)\n" + "\n".join(sub_trace))
        return pool

    # ------------------------------------------------------------------
    # Step 7: 2-hop typed graph traversal
    # ------------------------------------------------------------------

    def _graph_traverse(
        self,
        direction_key: str,
        matched_entity: dict[str, Any] | None,
        instr_emb: list[float] | None,
        trace: list[str],
    ) -> list[dict[str, Any]]:
        """Hop1+hop2 walk → enriched claim rows с provenance (path)."""
        if not self.graph_enabled:
            trace.append("## 7. Graph traversal\n- disabled (config)")
            return []
        if matched_entity is None:
            trace.append("## 7. Graph traversal\n- skipped: no matched entity")
            return []

        seed_id = int(matched_entity["id"])
        walk_rows = self._sql_graph_traverse_typed(
            direction_key=direction_key,
            seed_id=seed_id,
            max_hops=self.graph_max_hops,
            min_confidence=self.graph_min_confidence,
            hop1_limit=self.graph_hop1_limit,
            hop2_limit=self.graph_hop2_limit,
        )

        if not walk_rows:
            trace.append(
                "## 7. Graph traversal\n"
                f"- seed: id={seed_id}  ({matched_entity['canonical_name']!r})\n"
                f"- min_confidence: {self.graph_min_confidence}\n"
                "- hop1: 0  hop2: 0  → empty graph (изолированная сущность?)"
            )
            return []

        hop1_rows = [w for w in walk_rows if w["hop"] == 1]
        hop2_rows = [w for w in walk_rows if w["hop"] == 2]
        all_claim_ids = [w["claim_id"] for w in walk_rows]

        # Enrich: достаём полные строки claims с similarity к инструкции
        enriched = self._sql_fetch_claims_by_ids(direction_key, all_claim_ids, instr_emb)
        # Аннотируем path и hop из walk_rows
        walk_index = {w["claim_id"]: w for w in walk_rows}
        leaf_ids = list({w["leaf_entity_id"] for w in walk_rows if w.get("leaf_entity_id")})
        leaf_names = self._sql_fetch_entity_names(direction_key, leaf_ids)
        for row in enriched:
            w = walk_index.get(row["id"])
            if w is not None:
                row["graph_hop"] = w["hop"]
                row["graph_path"] = list(w["path"] or [])
                row["graph_leaf_id"] = w.get("leaf_entity_id")
                row["graph_leaf_name"] = leaf_names.get(w.get("leaf_entity_id"))
            row["source"] = "graph"

        sample_paths = [
            f"claim:{r['id']} via {r.get('graph_path', [])} → leaf={r.get('graph_leaf_name')!r}"
            for r in enriched[:4]
        ]
        header = (
            "## 7. Graph traversal\n"
            f"- seed: id={seed_id}  ({matched_entity['canonical_name']!r})\n"
            f"- min_confidence: {self.graph_min_confidence}\n"
            f"- hop1: {len(hop1_rows)}  hop2: {len(hop2_rows)}  total: {len(enriched)}"
        )
        if sample_paths:
            trace.append(header + "\n- sample paths:\n  " + "\n  ".join(sample_paths))
        else:
            trace.append(header)
        return enriched

    # ------------------------------------------------------------------
    # Step 8: Multi-signal re-rank
    # ------------------------------------------------------------------

    def _rerank(
        self,
        pool: dict[str, list[dict[str, Any]]],
        now: datetime,
        trace: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Считает score для каждой строки, сортирует по убыванию final, обрезает до top-K."""

        def score_claims(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
            for r in rows:
                comp = _claim_score(
                    r,
                    w_sim=self.claim_w_sim,
                    w_conf=self.claim_w_confidence,
                    w_tier=self.claim_w_tier,
                    w_recency=self.claim_w_recency,
                    half_life_days=self.recency_half_life_days,
                    now=now,
                )
                r["_score"] = comp
            rows.sort(key=lambda r: r["_score"]["final"], reverse=True)
            return rows[:top_k]

        def score_pages(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
            for r in rows:
                comp = _page_score(
                    r,
                    w_sim=self.page_w_sim,
                    w_quality=self.page_w_quality,
                    w_source_bonus=self.page_w_source_bonus,
                )
                r["_score"] = comp
            rows.sort(key=lambda r: r["_score"]["final"], reverse=True)
            return rows[:top_k]

        ranked = {
            "claims_entity": score_claims(
                list(pool["claims_entity"]), self.top_k_entity_biased_claims
            ),
            "claims_general": score_claims(
                list(pool["claims_general"]), self.top_k_general_claims
            ),
            "pages_general": score_pages(
                list(pool["pages_general"]), self.top_k_pages
            ),
        }

        sub_trace_lines: list[str] = ["### Top entity-biased claims:"]
        for r in ranked["claims_entity"][:5]:
            s = r["_score"]
            sub_trace_lines.append(
                f"- claim:{r['id']}  final={s['final']:.3f}  "
                f"(sim={s['sim']:.2f} conf={s['conf']:.2f} "
                f"tier={s['tier_s']:.2f} rec={s['recency_s']:.2f})"
            )
        sub_trace_lines.append("### Top general claims:")
        for r in ranked["claims_general"][:5]:
            s = r["_score"]
            sub_trace_lines.append(
                f"- claim:{r['id']}  final={s['final']:.3f}  "
                f"(sim={s['sim']:.2f} conf={s['conf']:.2f} "
                f"tier={s['tier_s']:.2f} rec={s['recency_s']:.2f})"
            )
        sub_trace_lines.append("### Top pages:")
        for r in ranked["pages_general"][:5]:
            s = r["_score"]
            sub_trace_lines.append(
                f"- [[{r['slug']}]] (kind={r['page_kind']})  final={s['final']:.3f}  "
                f"(sim={s['sim']:.2f} quality={s['quality']:.2f} "
                f"source={s['is_source']:.0f})"
            )

        trace.append(
            "## 8. Re-rank (multi-signal)\n"
            f"weights claims: sim={self.claim_w_sim} conf={self.claim_w_confidence} "
            f"tier={self.claim_w_tier} rec={self.claim_w_recency} "
            f"(half_life={self.recency_half_life_days}d)\n"
            f"weights pages: sim={self.page_w_sim} quality={self.page_w_quality} "
            f"source_bonus={self.page_w_source_bonus}\n\n"
            + "\n".join(sub_trace_lines)
        )
        return ranked

    # ------------------------------------------------------------------
    # Step 9: Dedup
    # ------------------------------------------------------------------

    def _dedup(
        self,
        *,
        direction_key: str,
        pool: dict[str, list[dict[str, Any]]],
        ranked: dict[str, list[dict[str, Any]]],
        trace: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Возвращает (final_pages, final_claims, final_chunks) с дедупом."""

        # Pages: entity_page → ranked pages_general (минус её id)
        final_pages: list[dict[str, Any]] = []
        seen_page_ids: set[int] = set()
        if pool["entity_page"]:
            ep = pool["entity_page"][0]
            final_pages.append(ep)
            seen_page_ids.add(int(ep["id"]))
        for p in ranked["pages_general"]:
            if int(p["id"]) in seen_page_ids:
                continue
            final_pages.append(p)
            seen_page_ids.add(int(p["id"]))

        # Claims: entity_biased первыми, потом general (включая graph),
        # с дедупом по id
        final_claims: list[dict[str, Any]] = []
        seen_claim_ids: set[int] = set()
        for c in ranked["claims_entity"]:
            cid = int(c["id"])
            if cid in seen_claim_ids:
                continue
            final_claims.append(c)
            seen_claim_ids.add(cid)
        for c in ranked["claims_general"]:
            cid = int(c["id"])
            if cid in seen_claim_ids:
                continue
            final_claims.append(c)
            seen_claim_ids.add(cid)

        # Chunks: entity-mentions первыми, дальше chunks для финальных claims
        final_chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[int] = set()
        for ch in pool["chunks_entity"]:
            cid = int(ch["chunk_id"])
            if cid in seen_chunk_ids:
                continue
            final_chunks.append(ch)
            seen_chunk_ids.add(cid)

        # Дотягиваем chunks для финальных claim-ов
        if final_claims:
            claim_chunk_rows = self._sql_chunks_for_claims(
                direction_key=direction_key,
                claim_ids=[int(c["id"]) for c in final_claims],
            )
            # Cap по top_k_chunks_per_claim для всего набора
            count_added = 0
            for ch in claim_chunk_rows:
                if count_added >= self.top_k_chunks_per_claim * max(1, len(final_claims)):
                    break
                cid = int(ch["chunk_id"])
                if cid in seen_chunk_ids:
                    continue
                final_chunks.append(ch)
                seen_chunk_ids.add(cid)
                count_added += 1

        # Жёсткий cap чанков, чтобы не разрывать budget
        max_total_chunks = max(
            self.top_k_chunks_per_entity + self.top_k_chunks_per_claim,
            12,
        )
        final_chunks = final_chunks[:max_total_chunks]

        sub_trace = [
            f"- final pages: {len(final_pages)} ({[p['slug'] for p in final_pages]})",
            f"- final claims: {len(final_claims)}",
            f"- final chunks: {len(final_chunks)}",
        ]
        # Inserting at right position — но выше у нас уже step 8, так что
        # Dedup присоединяется к step 8 как подсекция, чтобы трасса не разрослась.
        trace[-1] += "\n\n### Dedup\n" + "\n".join(sub_trace)
        return final_pages, final_claims, final_chunks

    # ------------------------------------------------------------------
    # Step 10: Contradiction arbiter
    # ------------------------------------------------------------------

    def _arbitrate_contradictions(
        self,
        direction_key: str,
        final_claims: list[dict[str, Any]],
        trace: list[str],
    ) -> list[dict[str, Any]]:
        """Для финального claim-set находит пары из rag_v7.claim_contradictions
        и для каждой запускает LLM-арбитра."""

        if not final_claims:
            trace.append("## 9. Contradiction arbiter\n- skipped: no final claims")
            return []

        claim_ids = [int(c["id"]) for c in final_claims]
        pairs = self._sql_contradiction_pairs(direction_key, claim_ids)
        if not pairs:
            trace.append(
                "## 9. Contradiction arbiter\n- pairs found: 0 (нет открытых "
                "противоречий в финальном claim-set)"
            )
            return []

        # Подгружаем строки claim'ов, которых ещё нет в final_claims
        by_id = {int(c["id"]): c for c in final_claims}
        missing_ids = [
            cid for p in pairs for cid in (p["claim_a_id"], p["claim_b_id"])
            if cid not in by_id
        ]
        if missing_ids:
            extra = self._sql_fetch_claims_by_ids(direction_key, list(set(missing_ids)), None)
            for r in extra:
                by_id[int(r["id"])] = r

        hints: list[dict[str, Any]] = []
        model_name = getattr(self.llm, "model_name", "unknown")
        sub_lines: list[str] = [f"- pairs found: {len(pairs)} (model: {model_name})"]

        for pair in pairs[: self.contradiction_arbiter_max_pairs]:
            a = by_id.get(int(pair["claim_a_id"]))
            b = by_id.get(int(pair["claim_b_id"]))
            if a is None or b is None:
                continue
            prompt = self._build_contradiction_prompt(a, b)
            try:
                decision = self.llm.structured(
                    _CONTRADICTION_ARBITER_SYSTEM,
                    prompt,
                    _ContradictionArbiterResponse,
                )
            except Exception as exc:  # noqa: BLE001
                sub_lines.append(
                    f"  - pair claim:{a['id']} vs claim:{b['id']}: "
                    f"**error** {exc.__class__.__name__}"
                )
                continue
            valid_ids = {int(a["id"]), int(b["id"])}
            picked = decision.likely_correct_claim_id
            if picked is not None and picked not in valid_ids:
                picked = None
            hint = {
                "claim_a_id": int(a["id"]),
                "claim_b_id": int(b["id"]),
                "likely_correct_claim_id": picked,
                "reasoning": decision.reasoning[:400],
                "confidence": float(decision.confidence),
            }
            hints.append(hint)
            sub_lines.append(
                f"  - pair claim:{a['id']} vs claim:{b['id']}  →  "
                f"hint: {('claim:' + str(picked)) if picked else 'null'}  "
                f"(conf={decision.confidence:.2f})"
            )

        if len(pairs) > self.contradiction_arbiter_max_pairs:
            sub_lines.append(
                f"  - (capped at {self.contradiction_arbiter_max_pairs}; "
                f"{len(pairs) - self.contradiction_arbiter_max_pairs} pairs not arbitered)"
            )

        trace.append("## 9. Contradiction arbiter\n" + "\n".join(sub_lines))
        return hints

    @staticmethod
    def _build_contradiction_prompt(
        a: dict[str, Any], b: dict[str, Any]
    ) -> str:
        def _claim_block(label: str, c: dict[str, Any]) -> str:
            return (
                f"{label}:\n"
                f"  id: {c['id']}\n"
                f"  text: {c.get('claim_text', '')!r}\n"
                f"  subject: {c.get('subject_name', '?')!r}\n"
                f"  predicate: {c.get('predicate', '?')!r}\n"
                f"  object: {c.get('object_repr', '?')!r}\n"
                f"  confidence: {c.get('confidence')}\n"
                f"  times_confirmed: {c.get('times_confirmed')}\n"
                f"  tier: {c.get('tier')}\n"
                f"  status: {c.get('status')}\n"
                f"  last_confirmed_at: {c.get('last_confirmed_at')}"
            )

        return (
            _claim_block("CLAIM A", a)
            + "\n\n"
            + _claim_block("CLAIM B", b)
            + "\n\nКакой из них вероятнее? (или null, если данных недостаточно)"
        )

    # ------------------------------------------------------------------
    # Step 11: Compose context
    # ------------------------------------------------------------------

    def _compose_context(
        self,
        *,
        position_or_role: str,
        instruction: str,
        matched_entity: dict[str, Any] | None,
        pages: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        contradiction_hints: list[dict[str, Any]],
        trace: list[str],
    ) -> tuple[str, dict[str, list[Any]]]:
        budget = self.max_context_chars
        parts: list[str] = []
        included: dict[str, list[Any]] = {
            "page_ids": [],
            "page_slugs": [],
            "claim_ids": [],
            "chunk_ids": [],
        }

        def _consume(text: str) -> None:
            nonlocal budget
            if budget <= 0:
                return
            if len(text) > budget:
                text = text[:budget].rstrip() + "\n…(budget cut)"
            parts.append(text)
            budget -= len(text)

        _consume(
            "QUESTION CONTEXT — НЕ ОТВЕЧАЙ НА INSTRUCTION, СОБЕРИ КОНТЕКСТ\n"
            f"POSITION_OR_ROLE: {position_or_role}\n"
            f"INSTRUCTION (для которой собирается контекст; НЕ отвечай на неё): {instruction}\n"
        )

        # PRIMARY ENTITY
        if matched_entity is not None:
            attrs = self._compact_json(matched_entity.get("salient_attrs"))
            if len(attrs) > 400:
                attrs = attrs[:400] + "…"
            _consume(
                "\nPRIMARY ENTITY:\n"
                f"  id={matched_entity['id']}, type={matched_entity['entity_type']}, "
                f"name={matched_entity['canonical_name']!r}, "
                f"mentions={matched_entity.get('mention_count', 0)}\n"
                f"  salient_attrs: {attrs}\n"
            )
        else:
            _consume("\nPRIMARY ENTITY:\n  (не определена; pipeline шёл без entity-bias)\n")

        # ENTITY WIKI PAGE (если есть, это первая в pages)
        entity_page: dict[str, Any] | None = None
        general_pages: list[dict[str, Any]] = list(pages)
        if matched_entity is not None and pages:
            cand = pages[0]
            if cand.get("page_kind") == "entity":
                entity_page = cand
                general_pages = pages[1:]

        if entity_page is not None and budget > 0:
            content = (entity_page.get("content_md") or "").strip()
            if len(content) > self.max_chars_per_page:
                content = content[: self.max_chars_per_page].rstrip() + "\n…(truncated)"
            quality = entity_page.get("quality_score")
            _consume(
                "\nENTITY WIKI PAGE:\n"
                f"### [[{entity_page['title']}]] (slug={entity_page['slug']}, kind=entity, "
                f"quality={quality})\n{content}\n"
            )
            included["page_ids"].append(entity_page["id"])
            included["page_slugs"].append(entity_page["slug"])

        # RELEVANT PAGES
        if general_pages and budget > 0:
            _consume("\nRELEVANT PAGES:\n")
            for p in general_pages:
                if budget <= 0:
                    break
                content = (p.get("content_md") or "").strip()
                if len(content) > self.max_chars_per_page:
                    content = content[: self.max_chars_per_page].rstrip() + "\n…(truncated)"
                sim = p.get("similarity")
                sim_s = f" sim={float(sim):.2f}" if sim is not None else ""
                quality = p.get("quality_score")
                score = (p.get("_score") or {}).get("final")
                score_s = f" score={float(score):.2f}" if score is not None else ""
                _consume(
                    f"\n### [[{p['title']}]] (slug={p['slug']}, kind={p['page_kind']},"
                    f"{sim_s} quality={quality}{score_s})\n{content}\n"
                )
                included["page_ids"].append(p["id"])
                included["page_slugs"].append(p["slug"])

        # ENTITY-BIASED + GENERAL CLAIMS (две подсекции)
        # Различаем по наличию graph_path (graph-derived) vs обычные.
        regular_claims = [c for c in claims if not c.get("graph_path")]
        graph_in_claims = [c for c in claims if c.get("graph_path")]

        if regular_claims and budget > 0:
            _consume("\nRELEVANT CLAIMS:\n")
            for c in regular_claims:
                if budget <= 0:
                    break
                line = self._format_claim_line(c)
                _consume(line + "\n")
                included["claim_ids"].append(int(c["id"]))

        if graph_in_claims and budget > 0:
            _consume("\nGRAPH-DERIVED CLAIMS (path: matched → … → leaf):\n")
            for c in graph_in_claims:
                if budget <= 0:
                    break
                path = " → ".join(c.get("graph_path") or [])
                leaf = c.get("graph_leaf_name") or c.get("graph_leaf_id")
                line = (
                    f"- claim:{c['id']}  via [{path}]  →  leaf={leaf!r}  "
                    f"({c.get('subject_name', '?')} {c.get('predicate', '?')} "
                    f"{c.get('object_repr', '?')})  "
                    f"(conf={float(c.get('confidence') or 0):.2f}, tier={c.get('tier')})"
                )
                _consume(line + "\n")
                included["claim_ids"].append(int(c["id"]))

        # CONTRADICTION HINTS
        if contradiction_hints and budget > 0:
            _consume("\nCONTRADICTION HINTS (от LLM-арбитра; оба claim'а в выборке выше):\n")
            for h in contradiction_hints:
                if budget <= 0:
                    break
                picked = h.get("likely_correct_claim_id")
                hint_str = (
                    f"claim:{picked} более вероятен" if picked else "решение не определено"
                )
                line = (
                    f"- pair: claim:{h['claim_a_id']} vs claim:{h['claim_b_id']}  →  "
                    f"hint: {hint_str} (conf={h.get('confidence', 0):.2f})\n"
                    f"  reasoning: {h.get('reasoning', '')}"
                )
                _consume(line + "\n")

        # SOURCE EXCERPTS (raw)
        if chunks and budget > 0:
            _consume("\nSOURCE EXCERPTS (raw, приоритет над пересказами выше):\n")
            citations_per_chunk: dict[int, list[int]] = {}
            for ch in chunks:
                if "claim_id" in ch:
                    citations_per_chunk.setdefault(int(ch["chunk_id"]), []).append(
                        int(ch["claim_id"])
                    )
            seen_chunks_local: set[int] = set()
            for ch in chunks:
                if budget <= 0:
                    break
                cid = int(ch["chunk_id"])
                if cid in seen_chunks_local:
                    continue
                seen_chunks_local.add(cid)
                content = (ch.get("content") or "").strip()
                if len(content) > self.max_chars_per_chunk:
                    content = content[: self.max_chars_per_chunk].rstrip() + "\n…(truncated)"
                cited_by = ", ".join(
                    f"claim:{c}" for c in citations_per_chunk.get(cid, [])
                ) or ("via entity_mentions" if "extracted_form" in ch else "")
                doc = ch.get("document_id")
                _consume(
                    f"\n### chunk:{cid} (doc={doc}; {cited_by})\n{content}\n"
                )
                included["chunk_ids"].append(cid)

        text = "".join(parts)
        trace.append(
            "## 10. Context composition\n"
            f"- final size: {len(text)} chars / budget {self.max_context_chars}\n"
            f"- included pages: {included['page_slugs']}\n"
            f"- included claims: {included['claim_ids']}\n"
            f"- included chunks: {included['chunk_ids']}"
        )
        return text, included

    @staticmethod
    def _format_claim_line(c: dict[str, Any]) -> str:
        conf = float(c.get("confidence") or 0)
        times = int(c.get("times_confirmed") or 1)
        tier = c.get("tier", "?")
        status = c.get("status", "active")
        score = (c.get("_score") or {}).get("final")
        score_s = f" score={float(score):.2f}" if score is not None else ""
        line = (
            f"- claim:{c['id']}  {c.get('subject_name', '?')} → "
            f"{c.get('predicate', '?')} → {c.get('object_repr', '?')}  "
            f"(×{times}, conf={conf:.2f}, {tier}{score_s})"
        )
        if status == "flagged_contradiction":
            line += "  ⚠ flagged_contradiction"
        return line

    # ------------------------------------------------------------------
    # Step 12: Synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        *,
        position_or_role: str,
        instruction: str,
        context_text: str,
        trace: list[str],
    ) -> _KnowledgeContextResponse:
        # Дополнительное предупреждение в user_prompt — четвёртый слой защиты
        # «не отвечай на instruction».
        user_prompt = (
            context_text
            + "\n\n"
            + "POSITION_OR_ROLE: " + position_or_role + "\n"
            + "INSTRUCTION (для которой собирается контекст; НЕ отвечай на неё): "
            + instruction + "\n\n"
            + "Сформируй context_summary, key_facts (с ссылками [[slug]]/claim:N/chunk:N) "
            + "и coverage_notes. НЕ отвечай на instruction."
        )
        model_name = getattr(self.llm, "model_name", "unknown")
        synth = self.llm.structured(
            _KNOWLEDGE_SYNTHESIS_SYSTEM, user_prompt, _KnowledgeContextResponse
        )
        trace.append(
            "## 11. Synthesis\n"
            f"- model: {model_name}\n"
            f"- prompt chars: {len(user_prompt)}\n"
            f"- summary chars: {len(synth.context_summary)}\n"
            f"- key_facts: {len(synth.key_facts)}\n"
            f"- cited slugs: {synth.cited_page_slugs}\n"
            f"- cited claim ids: {synth.cited_claim_ids}\n"
            f"- cited chunk ids: {synth.cited_chunk_ids}"
        )
        return synth

    # ------------------------------------------------------------------
    # Step 13: Format answer
    # ------------------------------------------------------------------

    @staticmethod
    def _format_answer(synth: _KnowledgeContextResponse) -> str:
        parts: list[str] = []
        if synth.context_summary.strip():
            parts.append(synth.context_summary.strip())
        if synth.key_facts:
            parts.append("\n### Key facts")
            for f in synth.key_facts:
                f = f.strip()
                if not f:
                    continue
                parts.append(f"- {f}" if not f.startswith("-") else f)
        if synth.coverage_notes.strip():
            parts.append("\n### Coverage")
            parts.append(synth.coverage_notes.strip())
        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # Step 14: Sources used + trace formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources_used(
        synth: _KnowledgeContextResponse,
        included: dict[str, list[Any]],
    ) -> str:
        # Используем то, на что синтезатор фактически сослался; fallback — все
        # включённые в контекст.
        slugs = synth.cited_page_slugs or included.get("page_slugs") or []
        claim_ids = synth.cited_claim_ids or included.get("claim_ids") or []
        chunk_ids = synth.cited_chunk_ids or included.get("chunk_ids") or []
        return (
            "## 12. Sources used\n"
            f"- pages: {[f'[[{s}]]' for s in slugs]}\n"
            f"- claims: {[f'claim:{c}' for c in claim_ids]}\n"
            f"- chunks: {[f'chunk:{c}' for c in chunk_ids]}"
        )

    @staticmethod
    def _format_trace(trace_steps: list[str], elapsed: float) -> str:
        return "\n\n".join(trace_steps) + f"\n\n## Elapsed: {elapsed:.2f}s"

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_json(value: Any) -> str:
        if value is None:
            return "{}"
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = ", ".join(f"{k}={v!r}" for k, v in value.items())
            return "{" + items + "}"
        return repr(value)

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    def _sql_check_direction_exists(self, direction_key: str) -> bool:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM rag_v7.directions WHERE key = %s LIMIT 1;",
                (direction_key,),
            )
            return cur.fetchone() is not None

    def _sql_top_entity_candidates(
        self,
        direction_key: str,
        role_emb: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        q = _to_vec(role_emb)
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
                (q, direction_key, q, k),
            )
            return cur.fetchall()

    def _sql_fetch_entity_by_id(
        self,
        direction_key: str,
        entity_id: int,
    ) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity_type, canonical_name, salient_attrs,
                       mention_count, confidence
                FROM rag_v7.entities
                WHERE id = %s AND direction_key = %s;
                """,
                (entity_id, direction_key),
            )
            return cur.fetchone()

    def _sql_wiki_page_for_entity(
        self,
        direction_key: str,
        entity_id: int,
    ) -> dict[str, Any] | None:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_kind::text AS page_kind, slug, title, content_md,
                       quality_score, coverage_claims, coverage_contradictions
                FROM rag_v7.wiki_pages
                WHERE direction_key = %s
                  AND entity_id = %s
                  AND page_kind = 'entity'::rag_v7.wiki_page_kind;
                """,
                (direction_key, entity_id),
            )
            return cur.fetchone()

    def _sql_top_claims_about_entity(
        self,
        direction_key: str,
        entity_id: int,
        instr_emb: list[float],
        pool_size: int,
    ) -> list[dict[str, Any]]:
        q = _to_vec(instr_emb)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.predicate, c.claim_text, c.confidence,
                       c.times_confirmed, c.tier::text AS tier, c.status::text AS status,
                       c.last_confirmed_at, c.subject_entity_id, c.object_entity_id,
                       e1.canonical_name AS subject_name,
                       COALESCE(e2.canonical_name, c.object_text) AS object_repr,
                       1 - (c.claim_embedding <=> %s) AS similarity
                FROM rag_v7.claims c
                JOIN rag_v7.entities e1 ON e1.id = c.subject_entity_id
                LEFT JOIN rag_v7.entities e2 ON e2.id = c.object_entity_id
                WHERE c.direction_key = %s
                  AND c.status::text = ANY(%s)
                  AND (c.subject_entity_id = %s OR c.object_entity_id = %s)
                ORDER BY c.claim_embedding <=> %s
                LIMIT %s;
                """,
                (
                    q,
                    direction_key,
                    list(_ACTIVE_STATUSES),
                    entity_id,
                    entity_id,
                    q,
                    pool_size,
                ),
            )
            return cur.fetchall()

    def _sql_top_claims_general(
        self,
        direction_key: str,
        instr_emb: list[float],
        pool_size: int,
    ) -> list[dict[str, Any]]:
        q = _to_vec(instr_emb)
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.predicate, c.claim_text, c.confidence,
                       c.times_confirmed, c.tier::text AS tier, c.status::text AS status,
                       c.last_confirmed_at, c.subject_entity_id, c.object_entity_id,
                       e1.canonical_name AS subject_name,
                       COALESCE(e2.canonical_name, c.object_text) AS object_repr,
                       1 - (c.claim_embedding <=> %s) AS similarity
                FROM rag_v7.claims c
                JOIN rag_v7.entities e1 ON e1.id = c.subject_entity_id
                LEFT JOIN rag_v7.entities e2 ON e2.id = c.object_entity_id
                WHERE c.direction_key = %s
                  AND c.status::text = ANY(%s)
                ORDER BY c.claim_embedding <=> %s
                LIMIT %s;
                """,
                (q, direction_key, list(_ACTIVE_STATUSES), q, pool_size),
            )
            return cur.fetchall()

    def _sql_top_pages_general(
        self,
        direction_key: str,
        instr_emb: list[float],
        pool_size: int,
    ) -> list[dict[str, Any]]:
        q = _to_vec(instr_emb)
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
                (q, direction_key, self.include_page_kinds, q, pool_size),
            )
            return cur.fetchall()

    def _sql_chunks_for_entity(
        self,
        direction_key: str,
        entity_id: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT ch.id AS chunk_id, ch.content, ch.document_id, ch.ord,
                       em.extracted_form
                FROM rag_v7.entity_mentions em
                JOIN rag_v7.chunks ch ON ch.id = em.chunk_id
                WHERE em.entity_id = %s
                  AND em.direction_key = %s
                ORDER BY ch.id
                LIMIT %s;
                """,
                (entity_id, direction_key, limit),
            )
            return cur.fetchall()

    def _sql_chunks_for_claims(
        self,
        direction_key: str,
        claim_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not claim_ids or not direction_key:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT cc.claim_id, ch.id AS chunk_id, ch.content,
                       ch.document_id, ch.ord
                FROM rag_v7.claim_citations cc
                JOIN rag_v7.chunks ch ON ch.id = cc.chunk_id
                WHERE cc.direction_key = %s AND cc.claim_id = ANY(%s)
                ORDER BY ch.id;
                """,
                (direction_key, claim_ids),
            )
            rows = cur.fetchall()
        # Re-rank по позиции claim_id в исходном списке
        rank = {cid: i for i, cid in enumerate(claim_ids)}
        rows.sort(key=lambda r: (rank.get(r["claim_id"], 1_000_000), r["chunk_id"]))
        return rows

    def _sql_graph_traverse_typed(
        self,
        *,
        direction_key: str,
        seed_id: int,
        max_hops: int,
        min_confidence: float,
        hop1_limit: int,
        hop2_limit: int,
    ) -> list[dict[str, Any]]:
        """Recursive CTE: hop1 (LIMITED) + hop2 (LIMITED)."""
        # Чтобы LIMIT работал per-hop, делаем два отдельных запроса:
        # 1) hop1 — простая выборка, LIMIT graph_hop1_limit
        # 2) hop2 — JOIN от hop1.leaf_entity_id, LIMIT graph_hop2_limit
        # Это проще и понятнее, чем рекурсивный CTE с per-hop ограничениями.
        with self._cm.conn() as conn, conn.cursor() as cur:
            # hop1
            cur.execute(
                """
                SELECT
                    c.id AS claim_id,
                    c.predicate,
                    CASE WHEN c.subject_entity_id = %s THEN c.object_entity_id
                         ELSE c.subject_entity_id END AS leaf_entity_id,
                    1 AS hop,
                    ARRAY[c.predicate]::text[] AS path,
                    c.confidence
                FROM rag_v7.claims c
                WHERE c.direction_key = %s
                  AND c.status::text = ANY(%s)
                  AND c.confidence >= %s
                  AND (c.subject_entity_id = %s OR c.object_entity_id = %s)
                  AND c.object_entity_id IS NOT NULL
                ORDER BY c.confidence DESC, c.id
                LIMIT %s;
                """,
                (
                    seed_id,
                    direction_key,
                    list(_ACTIVE_STATUSES),
                    min_confidence,
                    seed_id,
                    seed_id,
                    hop1_limit,
                ),
            )
            hop1_rows = cur.fetchall()

            if max_hops < 2 or not hop1_rows:
                return hop1_rows

            # hop2 — для каждого hop1 leaf достать его соседей (исключая seed)
            leaf_ids = [
                int(r["leaf_entity_id"])
                for r in hop1_rows
                if r["leaf_entity_id"] is not None
            ]
            seen_claim_ids = {int(r["claim_id"]) for r in hop1_rows}
            if not leaf_ids:
                return hop1_rows

            # Для path'а нам нужен predicate hop1 на пути к каждому leaf'у
            leaf_to_h1_predicate = {
                int(r["leaf_entity_id"]): r["predicate"]
                for r in hop1_rows
                if r["leaf_entity_id"] is not None
            }

            cur.execute(
                """
                SELECT
                    c.id AS claim_id,
                    c.predicate,
                    CASE WHEN c.subject_entity_id = ANY(%s) THEN c.object_entity_id
                         ELSE c.subject_entity_id END AS leaf_entity_id,
                    CASE WHEN c.subject_entity_id = ANY(%s) THEN c.subject_entity_id
                         ELSE c.object_entity_id END AS via_entity_id,
                    2 AS hop,
                    c.confidence
                FROM rag_v7.claims c
                WHERE c.direction_key = %s
                  AND c.status::text = ANY(%s)
                  AND c.confidence >= %s
                  AND (c.subject_entity_id = ANY(%s) OR c.object_entity_id = ANY(%s))
                  AND c.object_entity_id IS NOT NULL
                  AND NOT (c.subject_entity_id = %s AND c.object_entity_id = ANY(%s))
                  AND NOT (c.object_entity_id = %s AND c.subject_entity_id = ANY(%s))
                ORDER BY c.confidence DESC, c.id
                LIMIT %s;
                """,
                (
                    leaf_ids,
                    leaf_ids,
                    direction_key,
                    list(_ACTIVE_STATUSES),
                    min_confidence,
                    leaf_ids,
                    leaf_ids,
                    seed_id,
                    leaf_ids,
                    seed_id,
                    leaf_ids,
                    hop2_limit,
                ),
            )
            hop2_raw = cur.fetchall()

        # Собираем path = [hop1_predicate, hop2_predicate], дедуп по claim_id
        hop2_rows: list[dict[str, Any]] = []
        for r in hop2_raw:
            cid = int(r["claim_id"])
            if cid in seen_claim_ids:
                continue
            seen_claim_ids.add(cid)
            via = r.get("via_entity_id")
            h1_pred = leaf_to_h1_predicate.get(int(via)) if via is not None else None
            r["path"] = [h1_pred, r["predicate"]] if h1_pred else [r["predicate"]]
            # leaf_entity_id может оказаться seed_id — отфильтруем
            if r.get("leaf_entity_id") == seed_id:
                continue
            r.pop("via_entity_id", None)
            hop2_rows.append(r)

        return list(hop1_rows) + hop2_rows

    def _sql_fetch_claims_by_ids(
        self,
        direction_key: str,
        claim_ids: list[int],
        instr_emb: list[float] | None,
    ) -> list[dict[str, Any]]:
        if not claim_ids:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            if instr_emb is not None:
                q = _to_vec(instr_emb)
                cur.execute(
                    """
                    SELECT c.id, c.predicate, c.claim_text, c.confidence,
                           c.times_confirmed, c.tier::text AS tier,
                           c.status::text AS status, c.last_confirmed_at,
                           c.subject_entity_id, c.object_entity_id,
                           e1.canonical_name AS subject_name,
                           COALESCE(e2.canonical_name, c.object_text) AS object_repr,
                           1 - (c.claim_embedding <=> %s) AS similarity
                    FROM rag_v7.claims c
                    JOIN rag_v7.entities e1 ON e1.id = c.subject_entity_id
                    LEFT JOIN rag_v7.entities e2 ON e2.id = c.object_entity_id
                    WHERE c.direction_key = %s AND c.id = ANY(%s);
                    """,
                    (q, direction_key, claim_ids),
                )
            else:
                cur.execute(
                    """
                    SELECT c.id, c.predicate, c.claim_text, c.confidence,
                           c.times_confirmed, c.tier::text AS tier,
                           c.status::text AS status, c.last_confirmed_at,
                           c.subject_entity_id, c.object_entity_id,
                           e1.canonical_name AS subject_name,
                           COALESCE(e2.canonical_name, c.object_text) AS object_repr,
                           NULL::real AS similarity
                    FROM rag_v7.claims c
                    JOIN rag_v7.entities e1 ON e1.id = c.subject_entity_id
                    LEFT JOIN rag_v7.entities e2 ON e2.id = c.object_entity_id
                    WHERE c.direction_key = %s AND c.id = ANY(%s);
                    """,
                    (direction_key, claim_ids),
                )
            return cur.fetchall()

    def _sql_fetch_entity_names(
        self,
        direction_key: str,
        entity_ids: list[int],
    ) -> dict[int, str]:
        if not entity_ids:
            return {}
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_name
                FROM rag_v7.entities
                WHERE direction_key = %s AND id = ANY(%s);
                """,
                (direction_key, entity_ids),
            )
            return {int(r["id"]): r["canonical_name"] for r in cur.fetchall()}

    def _sql_contradiction_pairs(
        self,
        direction_key: str,
        claim_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not claim_ids:
            return []
        with self._cm.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, claim_a_id, claim_b_id, notes, status::text AS status
                FROM rag_v7.claim_contradictions
                WHERE direction_key = %s
                  AND status::text = 'open'
                  AND (claim_a_id = ANY(%s) OR claim_b_id = ANY(%s))
                ORDER BY id;
                """,
                (direction_key, claim_ids, claim_ids),
            )
            return cur.fetchall()
