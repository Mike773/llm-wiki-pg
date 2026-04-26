"""Авто-резолюция противоречий после ingest.

Когда supersession-арбитр вернул `decision='contradiction'`, оба claim-а
помечаются `flagged_contradiction`, и эта функция запускает второй проход
LLM, чтобы предложить «победителя» по recency / source-count / confidence.
Если уверенность арбитра ≥ threshold — проигравший уходит в `superseded`,
противоречие закрывается. Иначе остаётся flagged_contradiction для ручного
разбирательства.
"""

from __future__ import annotations

from typing import Any

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.claims import ClaimDAO
from rag_v7_wiki.protocols import LLM
from rag_v7_wiki.schemas import ContradictionResolutionDecision


CONTRADICTION_RESOLVE_SYSTEM = (
    "Ты — арбитр противоречий в knowledge base. Перед тобой два утверждения "
    "об одной и той же сущности с одним предикатом, у которых конфликтующие "
    "объекты. Твоя задача — выбрать, какое утверждение, скорее всего, верное, "
    "опираясь на:\n"
    "- recency: более свежее last_confirmed_at обычно весомее;\n"
    "- source-count: больше times_confirmed → больше доверие;\n"
    "- внутреннюю консистентность текста утверждения.\n"
    "Если данных недостаточно — отвечай 'unresolved'."
)


def _format_claim_card(label: str, claim: dict[str, Any]) -> str:
    obj = (
        claim.get("object_canonical_name")
        or claim.get("object_text")
        or "—"
    )
    return (
        f"[{label}] id={claim['id']}\n"
        f"  predicate: {claim['predicate']}\n"
        f"  object ({claim['object_kind']}): {obj}\n"
        f"  text: {claim['claim_text']}\n"
        f"  times_confirmed: {claim.get('times_confirmed', 1)}\n"
        f"  confidence: {claim.get('confidence', 0):.2f}\n"
        f"  first_seen: {claim.get('first_seen_at')}\n"
        f"  last_confirmed: {claim.get('last_confirmed_at')}"
    )


def resolve_contradiction(
    direction_key: str,
    contradiction_id: int,
    claim_a: dict[str, Any],
    claim_b: dict[str, Any],
    llm: LLM,
    claim_dao: ClaimDAO,
    config: WikiConfig,
) -> ContradictionResolutionDecision:
    """Запускает арбитра. При уверенном решении применяет supersession."""
    user = (
        f"{_format_claim_card('A', claim_a)}\n\n{_format_claim_card('B', claim_b)}"
    )
    decision = llm.structured(
        CONTRADICTION_RESOLVE_SYSTEM, user, ContradictionResolutionDecision
    )

    if decision.confidence < config.contradiction_auto_resolve_min_confidence:
        return decision
    if decision.winner not in {"a", "b"}:
        return decision

    if decision.winner == "a":
        winner_id, loser_id = claim_a["id"], claim_b["id"]
    else:
        winner_id, loser_id = claim_b["id"], claim_a["id"]

    claim_dao.supersede(
        direction_key=direction_key,
        old_claim_id=loser_id,
        new_claim_id=winner_id,
        reason=decision.reason or "auto-resolved contradiction",
        decided_by="auto_arbiter",
    )
    claim_dao.resolve_contradiction(
        contradiction_id=contradiction_id,
        winner_id=winner_id,
        notes=decision.reason or None,
    )
    return decision
