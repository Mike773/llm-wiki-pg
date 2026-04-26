"""Нормализация predicates в контролируемый словарь канонических связей.

Тот же паттерн, что entity-resolution: kNN по эмбеддингам существующих
canonical_predicates → если top1 ≥ threshold, переиспользуем; иначе LLM-арбитр
решает «same» (с уже существующим) или «new» (заводим новую запись).
"""

from __future__ import annotations

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.predicates import PredicateDAO
from rag_v7_wiki.protocols import LLM, Embedder
from rag_v7_wiki.schemas import PredicateResolutionDecision


PREDICATE_RESOLVE_SYSTEM = (
    "Ты решаешь, описывает ли новый предикат тот же тип отношения, что и один "
    "из перечисленных канонических предикатов. Учитывай синонимы, переходы между "
    "языками и активным/пассивным залогами. Если сомневаешься — отвечай 'new' и "
    "предложи короткое каноническое имя (без кавычек, в lowercase, латиницей "
    "по возможности, например 'depends_on', 'authored_by', 'uses')."
)


def _arbitrate(
    predicate_text: str,
    candidates: list[dict],
    llm: LLM,
) -> PredicateResolutionDecision:
    cand_lines = [
        f"[{i}] {c['canonical']} (sim={c.get('similarity', 0):.2f})"
        + (f" — {c['description']}" if c.get("description") else "")
        for i, c in enumerate(candidates)
    ]
    user = (
        f"Новый предикат:\n  {predicate_text}\n\n"
        "Канонические кандидаты:\n" + ("\n".join(cand_lines) or "(пусто)")
    )
    return llm.structured(PREDICATE_RESOLVE_SYSTEM, user, PredicateResolutionDecision)


def normalize_predicate(
    direction_key: str,
    predicate_text: str,
    embedder: Embedder,
    llm: LLM,
    predicate_dao: PredicateDAO,
    config: WikiConfig,
) -> int:
    """Возвращает canonical_predicate_id для данного predicate_text.

    Если порог сходства не достигнут, обращается к LLM-арбитру; либо
    переиспользует существующий канонический предикат (если арбитр сказал «same»),
    либо создаёт новый с предложенным каноническим именем.
    """
    if not predicate_text or not predicate_text.strip():
        # На всякий случай — пустой предикат не нормализуем.
        canonical = "_empty"
        existing = predicate_dao.get_by_canonical(direction_key, canonical)
        if existing:
            return existing["id"]
        [emb] = embedder.embed([canonical])
        return predicate_dao.upsert(direction_key, canonical, emb)

    [embedding] = embedder.embed([predicate_text])

    candidates = predicate_dao.find_similar(
        direction_key=direction_key,
        query_embedding=embedding,
        top_k=config.predicate_normalization_pool,
        threshold=config.predicate_normalization_threshold,
    )

    if candidates:
        # Самый близкий — переиспользуем без LLM-арбитра.
        winner = candidates[0]
        predicate_dao.bump_use(winner["id"])
        return winner["id"]

    # Достаём более широкий пул для арбитра (включая ниже threshold).
    wider = predicate_dao.find_similar(
        direction_key=direction_key,
        query_embedding=embedding,
        top_k=config.predicate_normalization_pool,
        threshold=0.0,
    )

    proposed_canonical: str | None = None
    if wider:
        decision = _arbitrate(predicate_text, wider, llm)
        if decision.decision == "same" and decision.matched_canonical:
            existing = predicate_dao.get_by_canonical(
                direction_key, decision.matched_canonical
            )
            if existing:
                predicate_dao.bump_use(existing["id"])
                return existing["id"]
        proposed_canonical = decision.proposed_canonical

    canonical = proposed_canonical or _slugify_predicate(predicate_text)

    existing = predicate_dao.get_by_canonical(direction_key, canonical)
    if existing:
        predicate_dao.bump_use(existing["id"])
        return existing["id"]

    new_id = predicate_dao.upsert(
        direction_key=direction_key,
        canonical=canonical,
        embedding=embedding,
        description=predicate_text,
    )
    predicate_dao.bump_use(new_id)
    return new_id


def _slugify_predicate(text: str) -> str:
    """Резервный канонический slug если LLM ничего не предложил."""
    s = text.lower().strip()
    out: list[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "_", "-"):
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "predicate"
