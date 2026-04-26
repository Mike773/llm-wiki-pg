from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from rag_v7_wiki import WikiConfig, WikiCore
from rag_v7_wiki.schemas import (
    ContradictionResolutionDecision,
    SupersessionDecision,
)
from tests._helpers import count as _count
from tests._helpers import insert_document
from tests.conftest import FakeEmbedder, FakeLLM

T = TypeVar("T", bound=BaseModel)


class ContradictoryLLM(FakeLLM):
    """LLM-стаб, который форсирует contradiction → auto-resolved (winner='b')."""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if schema is SupersessionDecision:
            self.calls.append(("structured", schema.__name__))
            return SupersessionDecision(  # type: ignore[return-value]
                decision="contradiction",
                reason="forced contradiction for test",
            )
        if schema is ContradictionResolutionDecision:
            self.calls.append(("structured", schema.__name__))
            return ContradictionResolutionDecision(  # type: ignore[return-value]
                winner="b",
                confidence=0.9,
                reason="newer claim wins",
            )
        return super().structured(system, user, schema)


class UnresolvedLLM(FakeLLM):
    """LLM-стаб: contradiction есть, но арбитр не уверен → флаг остаётся."""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if schema is SupersessionDecision:
            self.calls.append(("structured", schema.__name__))
            return SupersessionDecision(  # type: ignore[return-value]
                decision="contradiction",
                reason="forced contradiction for test",
            )
        if schema is ContradictionResolutionDecision:
            self.calls.append(("structured", schema.__name__))
            return ContradictionResolutionDecision(  # type: ignore[return-value]
                winner="unresolved",
                confidence=0.0,
                reason="no enough evidence",
            )
        return super().structured(system, user, schema)


def test_auto_resolved_contradiction_marks_superseded(clean_db: str) -> None:
    embedder = FakeEmbedder()
    llm = ContradictoryLLM()
    with WikiCore(db_dsn=clean_db, embedder=embedder, llm=llm, config=WikiConfig()) as wiki:
        wiki.ensure_direction("dir-a")
        doc1 = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
        wiki.process_document("dir-a", doc1)
        doc2 = insert_document(clean_db, "dir-a", "Alice also works on Test Concept.")
        wiki.process_document("dir-a", doc2)

    auto_decided = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claim_supersedes WHERE decided_by = 'auto_arbiter'",
    )
    assert auto_decided >= 1

    superseded = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claims WHERE status = 'superseded'",
    )
    assert superseded >= 1

    resolved = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claim_contradictions WHERE status = 'resolved'",
    )
    assert resolved >= 1


def test_unresolved_contradiction_stays_flagged(clean_db: str) -> None:
    embedder = FakeEmbedder()
    llm = UnresolvedLLM()
    with WikiCore(db_dsn=clean_db, embedder=embedder, llm=llm, config=WikiConfig()) as wiki:
        wiki.ensure_direction("dir-a")
        doc1 = insert_document(clean_db, "dir-a", "Alice works on Test Concept.")
        wiki.process_document("dir-a", doc1)
        doc2 = insert_document(clean_db, "dir-a", "Alice also works on Test Concept.")
        wiki.process_document("dir-a", doc2)

    auto_decided = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claim_supersedes WHERE decided_by = 'auto_arbiter'",
    )
    assert auto_decided == 0

    flagged = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claims WHERE status = 'flagged_contradiction'",
    )
    assert flagged >= 2

    open_contradictions = _count(
        clean_db,
        "SELECT count(*) FROM rag_v7.claim_contradictions WHERE status = 'open'",
    )
    assert open_contradictions >= 1
