from __future__ import annotations

import psycopg

from rag_v7_wiki import WikiCore
from tests._helpers import insert_document


def test_log_and_index_pages_rendered_into_postgres(
    wiki: WikiCore, clean_db: str
) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(
        clean_db, "dir-a", "Alice works on Test Concept."
    )
    wiki.process_document("dir-a", doc_id)

    index_md = wiki.get_index_md("dir-a")
    log_md = wiki.get_log_md("dir-a")

    assert index_md is not None
    assert log_md is not None

    # Index должен содержать отсылки к entity-страницам и source-странице.
    assert "Index — dir-a" in index_md
    # Минимум один заголовок секции (Сущности или Источники).
    assert "##" in index_md

    # Log должен иметь хотя бы одну запись с маркером даты и kind=ingest.
    assert "Log — dir-a" in log_md
    assert "ingest" in log_md.lower()


def test_log_grows_on_each_ingest(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")

    doc1 = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
    wiki.process_document("dir-a", doc1)
    entries_after_first = wiki.list_log_entries("dir-a")
    assert len(entries_after_first) == 1
    assert entries_after_first[0]["event_kind"] == "ingest"

    doc2 = insert_document(clean_db, "dir-a", "Bob also uses Test Concept.")
    wiki.process_document("dir-a", doc2)
    entries_after_second = wiki.list_log_entries("dir-a")
    assert len(entries_after_second) == 2

    # log-страница тоже отражает обе записи.
    log_md = wiki.get_log_md("dir-a")
    assert log_md is not None
    assert log_md.count("ingest") >= 2


def test_index_singleton_per_direction(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    doc1 = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
    wiki.process_document("dir-a", doc1)
    doc2 = insert_document(clean_db, "dir-a", "Alice also leads it.")
    wiki.process_document("dir-a", doc2)

    with psycopg.connect(clean_db, autocommit=True) as conn:
        cur = conn.execute(
            "SELECT count(*) FROM rag_v7.wiki_pages "
            "WHERE direction_key = 'dir-a' AND page_kind = 'index';"
        )
        assert cur.fetchone()[0] == 1
        cur = conn.execute(
            "SELECT count(*) FROM rag_v7.wiki_pages "
            "WHERE direction_key = 'dir-a' AND page_kind = 'log';"
        )
        assert cur.fetchone()[0] == 1
