from __future__ import annotations

from typing import TypeVar

import psycopg
from pydantic import BaseModel

from rag_v7_wiki import WikiCore, WikiQuery
from rag_v7_wiki.query import _SlugPickResponse, _WikiAnswer
from tests._helpers import insert_document
from tests.conftest import FakeEmbedder, FakeLLM

T = TypeVar("T", bound=BaseModel)


def test_query_embeddings_mode(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(
        clean_db, "dir-a", "Alice works on Test Concept."
    )
    wiki.process_document("dir-a", doc_id)

    embedder = FakeEmbedder()
    llm = FakeLLM()
    with WikiQuery(
        connection_string=clean_db,
        direction_key="dir-a",
        llm=llm,
        embedder=embedder,
        min_similarity=0.0,
    ) as q:
        result = q.ask("Кто такой Alice?")

    assert result["answer"] is not None
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()

    report = result["report"]
    assert report["mode"] == "embeddings"
    stage_names = [s["name"] for s in report["stages"]]
    assert "embed_query" in stage_names
    assert "retrieve_entities" in stage_names
    assert "retrieve_claims" in stage_names
    assert "retrieve_pages" in stage_names
    assert "compose_context" in stage_names
    assert "synthesize" in stage_names

    # хоть что-то нашлось — иначе тест бы скатился в пустое направление
    retrieve_pages = next(s for s in report["stages"] if s["name"] == "retrieve_pages")
    assert retrieve_pages["count"] >= 1

    # answer_meta заполнен
    assert "answer_meta" in report
    assert "confidence" in report["answer_meta"]


def test_query_wiki_only_mode(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(
        clean_db, "dir-a", "Alice works on Test Concept."
    )
    wiki.process_document("dir-a", doc_id)

    # Берём реальный slug source-страницы, чтобы PickerLLM вернул валидный.
    with psycopg.connect(clean_db, autocommit=True) as conn:
        row = conn.execute(
            "SELECT slug FROM rag_v7.wiki_pages "
            "WHERE direction_key = 'dir-a' AND page_kind = 'source' LIMIT 1"
        ).fetchone()
    assert row is not None, "source page must exist after ingest"
    source_slug = row[0]

    class PickerLLM(FakeLLM):
        def structured(self, system: str, user: str, schema: type[T]) -> T:
            if schema is _SlugPickResponse:
                self.calls.append(("structured", schema.__name__))
                return _SlugPickResponse(  # type: ignore[return-value]
                    relevant_slugs=[source_slug],
                    reasoning="picked the source page",
                )
            return super().structured(system, user, schema)

    embedder = FakeEmbedder()
    llm = PickerLLM()
    with WikiQuery(
        connection_string=clean_db,
        direction_key="dir-a",
        llm=llm,
        embedder=embedder,
        use_embeddings=False,
    ) as q:
        result = q.ask("Что у нас есть в источнике?")

    assert result["answer"] is not None
    report = result["report"]
    assert report["mode"] == "wiki_only"

    pick_stage = next(
        (s for s in report["stages"] if s["name"] == "wiki_only_pick"), None
    )
    assert pick_stage is not None
    assert pick_stage["slugs"] == [source_slug]

    pages_stage = next(
        (s for s in report["stages"] if s["name"] == "retrieve_pages"), None
    )
    assert pages_stage is not None
    assert pages_stage["count"] >= 1

    # В embeddings-режиме нет — embed_query на всякий случай не должно быть.
    assert "embed_query" not in [s["name"] for s in report["stages"]]


def test_query_empty_direction(clean_db: str) -> None:
    embedder = FakeEmbedder()
    llm = FakeLLM()

    # ensure_direction без ingest — в БД ничего нет.
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO rag_v7.directions (key, name) VALUES ('empty', 'empty');"
        )

    with WikiQuery(
        connection_string=clean_db,
        direction_key="empty",
        llm=llm,
        embedder=embedder,
    ) as q:
        result = q.ask("Любой вопрос")

    report = result["report"]

    # Все retrieve-стадии прошли, но с count=0.
    for name in ("retrieve_entities", "retrieve_claims", "retrieve_pages"):
        stage = next(s for s in report["stages"] if s["name"] == name)
        assert stage["count"] == 0, f"{name} should be empty for empty direction"

    # synthesize пропущен — ответ детерминированный.
    synth = next(s for s in report["stages"] if s["name"] == "synthesize")
    assert synth.get("skipped") is True
    assert synth.get("reason") == "empty_context"

    assert result["answer"] is not None  # детерминированное «недостаточно данных»
    assert "Недостаточно данных" in result["answer"]
    assert report["answer_meta"]["insufficient_evidence"] is True
    assert report["answer_meta"]["confidence"] == 0.0
