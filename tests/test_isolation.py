from __future__ import annotations

from rag_v7_wiki import WikiCore
from tests._helpers import count as _count
from tests._helpers import insert_document


def test_direction_isolation(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    wiki.ensure_direction("dir-b")

    doc_a = insert_document(clean_db, "dir-a", "Alice is in direction A.")
    doc_b = insert_document(clean_db, "dir-b", "Alice is in direction B (different person).")

    # Обрабатываем только direction A
    wiki.process_document("dir-a", doc_a)

    # Direction A должен иметь записи
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.entities WHERE direction_key = 'dir-a'",
    ) >= 1
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.wiki_pages WHERE direction_key = 'dir-a'",
    ) >= 1

    # Direction B должен оставаться пустым на всех дочерних таблицах
    for table in (
        "chunks",
        "entities",
        "claims",
        "wiki_pages",
        "page_links",
        "page_sources",
        "wiki_log_entries",
        "canonical_predicates",
    ):
        assert _count(
            clean_db,
            f"SELECT count(*) FROM rag_v7.{table} WHERE direction_key = 'dir-b'",
        ) == 0, f"direction-b leaked into {table}"

    # Документ в direction-b остался pending
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.documents WHERE id = %s AND status = 'pending'",
        (doc_b,),
    ) == 1


def test_process_pending_scoped(wiki: WikiCore, clean_db: str) -> None:
    wiki.ensure_direction("dir-a")
    wiki.ensure_direction("dir-b")
    doc_a = insert_document(clean_db, "dir-a", "Alice in A.")
    doc_b = insert_document(clean_db, "dir-b", "Alice in B.")

    wiki.process_pending("dir-a", limit=10)

    # Только doc_a обработан
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.documents WHERE id = %s AND status = 'processed'",
        (doc_a,),
    ) == 1
    assert _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.documents WHERE id = %s AND status = 'pending'",
        (doc_b,),
    ) == 1
