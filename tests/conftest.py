from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import psycopg
import pytest
from pydantic import BaseModel
from testcontainers.postgres import PostgresContainer

from rag_v7_wiki import WikiConfig, WikiCore
from rag_v7_wiki.schemas import (
    ClaimsResponse,
    ContradictionResolutionDecision,
    DocumentSummaryResponse,
    EntitiesResponse,
    ExtractedClaim,
    ExtractedEntity,
    PageQualityResponse,
    PageSynthesisResponse,
    PredicateResolutionDecision,
    ResolutionDecision,
    SourcePageResponse,
    SupersessionDecision,
)

T = TypeVar("T", bound=BaseModel)

MIGRATION = Path(__file__).resolve().parents[1] / "migrations" / "rag_v7_schema.sql"


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    container = PostgresContainer(
        "pgvector/pgvector:pg16",
        username="rag",
        password="rag",
        dbname="rag",
    )
    with container as pg:
        host = pg.get_container_host_ip()
        port = pg.get_exposed_port(5432)
        dsn = f"postgresql://rag:rag@{host}:{port}/rag"
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(MIGRATION.read_text())
        yield dsn


@pytest.fixture
def clean_db(pg_dsn: str) -> str:
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE rag_v7.directions CASCADE;")
    return pg_dsn


class FakeEmbedder:
    dim = 2560

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            extended = (digest * (self.dim // len(digest) + 1))[: self.dim]
            out.append([(b - 128) / 128.0 for b in extended])
        return out


class FakeLLM:
    """Минимальный стаб LLM. Возвращает фиксированные структурированные ответы."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def complete(self, system: str, user: str) -> str:
        self.calls.append(("complete", system))
        return ""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(("structured", schema.__name__))
        if schema is DocumentSummaryResponse:
            return DocumentSummaryResponse(summary=user[:200])  # type: ignore[return-value]
        if schema is EntitiesResponse:
            return EntitiesResponse(  # type: ignore[return-value]
                entities=[
                    ExtractedEntity(
                        entity_type="Concept",
                        canonical_name="Test Concept",
                        aliases=[],
                        salient_attrs={"role": "primary"},
                        supporting_chunk_indices=[0],
                    ),
                    ExtractedEntity(
                        entity_type="Person",
                        canonical_name="Alice",
                        aliases=["A."],
                        supporting_chunk_indices=[0],
                    ),
                ]
            )
        if schema is ResolutionDecision:
            return ResolutionDecision(decision="new")  # type: ignore[return-value]
        if schema is ClaimsResponse:
            return ClaimsResponse(  # type: ignore[return-value]
                claims=[
                    ExtractedClaim(
                        subject_canonical_name="Alice",
                        predicate="works on",
                        object_kind="entity",
                        object_canonical_name="Test Concept",
                        claim_text="Alice works on Test Concept.",
                        citation_chunk_indices=[0],
                        confidence_hint=0.8,
                    ),
                ]
            )
        if schema is SupersessionDecision:
            return SupersessionDecision(decision="orthogonal", reason="stub")  # type: ignore[return-value]
        if schema is PageSynthesisResponse:
            title = "Stub Page"
            return PageSynthesisResponse(  # type: ignore[return-value]
                title=title,
                content_md=(
                    f"# {title}\n\nStub-страница, ссылается на [[Test Concept]]."
                ),
            )
        if schema is SourcePageResponse:
            return SourcePageResponse(  # type: ignore[return-value]
                title="Stub Source",
                abstract="Stub-аннотация источника.",
                key_takeaways=["Stub takeaway"],
                top_entities=["Test Concept", "Alice"],
            )
        if schema is PredicateResolutionDecision:
            return PredicateResolutionDecision(  # type: ignore[return-value]
                decision="new",
                matched_canonical=None,
                proposed_canonical=None,
                reason="stub",
            )
        if schema is ContradictionResolutionDecision:
            return ContradictionResolutionDecision(  # type: ignore[return-value]
                winner="unresolved",
                confidence=0.0,
                reason="stub",
            )
        if schema is PageQualityResponse:
            return PageQualityResponse(  # type: ignore[return-value]
                score=0.9,
                issues=[],
                suggestions=[],
                needs_resynthesis=False,
            )
        raise NotImplementedError(f"FakeLLM не умеет {schema.__name__}")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def wiki(
    clean_db: str,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLLM,
) -> Iterator[WikiCore]:
    core = WikiCore(
        db_dsn=clean_db,
        embedder=fake_embedder,
        llm=fake_llm,
        config=WikiConfig(),
    )
    try:
        yield core
    finally:
        core.close()


