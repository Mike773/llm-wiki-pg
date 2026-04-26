from __future__ import annotations

from rag_v7_wiki import WikiCore
from tests._helpers import count as _count
from tests._helpers import insert_document


def test_canonical_predicate_created_on_ingest(
    wiki: WikiCore, clean_db: str
) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
    wiki.process_document("dir-a", doc_id)

    n = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.canonical_predicates WHERE direction_key = 'dir-a'",
    )
    assert n == 1

    used = _count(
        clean_db,
        "SELECT times_used FROM rag_v7.canonical_predicates "
        "WHERE direction_key = 'dir-a' LIMIT 1",
    )
    assert used >= 1


def test_canonical_predicate_reused_on_repeat(
    wiki: WikiCore, clean_db: str
) -> None:
    wiki.ensure_direction("dir-a")
    doc1 = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
    doc2 = insert_document(clean_db, "dir-a", "Bob also works on Test Concept.")
    wiki.process_document("dir-a", doc1)
    wiki.process_document("dir-a", doc2)

    # FakeEmbedder детерминирован, FakeLLM возвращает predicate='works on'
    # на оба документа → canonical_predicate должен переиспользоваться.
    n = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.canonical_predicates WHERE direction_key = 'dir-a'",
    )
    assert n == 1


def test_all_claims_have_canonical_predicate(
    wiki: WikiCore, clean_db: str
) -> None:
    wiki.ensure_direction("dir-a")
    doc_id = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
    wiki.process_document("dir-a", doc_id)

    total = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claims WHERE direction_key = 'dir-a'",
    )
    canonical = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claims "
        "WHERE direction_key = 'dir-a' AND canonical_predicate_id IS NOT NULL",
    )
    assert canonical == total
    assert total >= 1
