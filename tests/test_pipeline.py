from __future__ import annotations

import psycopg

from rag_v7_wiki import WikiCore
from tests._helpers import count as _count
from tests._helpers import insert_document


def test_smoke_pipeline_creates_artifacts(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a", name="Direction A")
    doc_id = insert_document(
        clean_db,
        direction_key="dir-a",
        content="Alice is the lead engineer working on the Test Concept project.",
    )

    wiki.process_document("dir-a", doc_id)

    assert _count(
        clean_db, "SELECT count(*) FROM rag_v7.chunks WHERE document_id = %s", (doc_id,)
    ) == 1
    assert _count(clean_db, "SELECT count(*) FROM rag_v7.entities") == 2
    assert _count(clean_db, "SELECT count(*) FROM rag_v7.claims WHERE status = 'active'") >= 1

    # Entity-pages: только subjects claim-ов получают страницы.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_pages WHERE page_kind = 'entity'",
    ) >= 1
    # Source-page для документа.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_pages WHERE page_kind = 'source' AND source_document_id = %s",
        (doc_id,),
    ) == 1
    # Singleton index/log.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_pages WHERE page_kind = 'index'",
    ) == 1
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_pages WHERE page_kind = 'log'",
    ) == 1

    # Лог-запись в структурной таблице.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_log_entries WHERE direction_key = 'dir-a' AND event_kind = 'ingest'",
    ) == 1

    # Canonical predicate появился.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.canonical_predicates WHERE direction_key = 'dir-a'",
    ) >= 1
    # На claim-е выставлен canonical_predicate_id.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claims WHERE direction_key = 'dir-a' AND canonical_predicate_id IS NOT NULL",
    ) >= 1

    # Provenance: page_sources заполнен для entity-страниц.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.page_sources WHERE direction_key = 'dir-a'",
    ) >= 1

    # Coverage / quality на entity-страницах.
    assert _count(
        clean_db,
        """
        SELECT count(*) FROM rag_v7.wiki_pages
        WHERE direction_key = 'dir-a' AND page_kind = 'entity'
          AND quality_score IS NOT NULL AND coverage_claims > 0
        """,
    ) >= 1

    assert _count(clean_db, "SELECT count(*) FROM rag_v7.wiki_page_revisions") >= 1
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.documents WHERE id = %s AND status = 'processed'",
        (doc_id,),
    ) == 1

    # Public API: index/log markdown доступны.
    assert wiki.get_index_md("dir-a") is not None
    assert wiki.get_log_md("dir-a") is not None
    assert wiki.get_source_page("dir-a", doc_id) is not None


def test_idempotent_reprocess(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(
        clean_db,
        direction_key="dir-a",
        content="Alice is the lead engineer working on the Test Concept project.",
    )
    wiki.process_document("dir-a", doc_id)
    chunks_before = _count(clean_db, "SELECT count(*) FROM rag_v7.chunks")
    entities_before = _count(clean_db, "SELECT count(*) FROM rag_v7.entities")
    pages_before = _count(clean_db, "SELECT count(*) FROM rag_v7.wiki_pages")
    log_entries_before = _count(
        clean_db, "SELECT count(*) FROM rag_v7.wiki_log_entries"
    )

    # Принудительно вернуть документ в pending и перезапустить
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute(
            "UPDATE rag_v7.documents SET status = 'pending' WHERE id = %s;",
            (doc_id,),
        )
    wiki.process_document("dir-a", doc_id)

    assert _count(clean_db, "SELECT count(*) FROM rag_v7.chunks") == chunks_before
    assert _count(clean_db, "SELECT count(*) FROM rag_v7.entities") == entities_before
    # Новых wiki-страниц не появилось — index/log/source/entity все upsert-ятся.
    assert _count(clean_db, "SELECT count(*) FROM rag_v7.wiki_pages") == pages_before
    # А вот log-entry должна добавиться: каждый успешный ingest пишет строку.
    assert (
        _count(clean_db, "SELECT count(*) FROM rag_v7.wiki_log_entries")
        == log_entries_before + 1
    )


def test_process_pending(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    doc1 = insert_document(clean_db, "dir-a", "Alice loves Test Concept.")
    doc2 = insert_document(clean_db, "dir-a", "Bob also touches Test Concept.")

    processed = wiki.process_pending("dir-a", limit=10)
    assert sorted(processed) == sorted([doc1, doc2])
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.documents WHERE status = 'processed'",
    ) == 2
    # Два ingest-события в логе.
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_log_entries WHERE direction_key = 'dir-a'",
    ) == 2
