from __future__ import annotations

from dataclasses import dataclass, field

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.claims import ClaimDAO
from rag_v7_wiki.dao.predicates import PredicateDAO
from rag_v7_wiki.pipeline.contradictions import resolve_contradiction
from rag_v7_wiki.pipeline.predicates import normalize_predicate
from rag_v7_wiki.protocols import LLM, Embedder
from rag_v7_wiki.schemas import (
    ClaimsResponse,
    ExtractedClaim,
    SupersessionDecision,
)

CLAIMS_SYSTEM = (
    "Ты извлекаешь атомарные утверждения (claims) из текста. Каждое утверждение — "
    "одна простая сущность-связь:\n"
    "- subject_canonical_name: канонические имя субъекта (должно совпадать с одной "
    "из перечисленных сущностей).\n"
    "- predicate: краткая глагольная фраза в свободной форме (без кавычек).\n"
    "- object_kind: 'entity' если объект — другая известная сущность, иначе 'literal'.\n"
    "- object_canonical_name: имя сущности-объекта (если object_kind='entity').\n"
    "- object_text: буквальный текст (если object_kind='literal').\n"
    "- claim_text: одно предложение в естественной форме.\n"
    "- citation_chunk_indices: индексы чанков (## Chunk N), откуда взято.\n"
    "- confidence_hint: насколько уверен от 0.0 до 1.0.\n"
    "Не выдумывай ничего, чего нет в тексте."
)


SUPERSESSION_SYSTEM = (
    "Ты сравниваешь два утверждения об одной и той же сущности с одним предикатом. "
    "Выбери одно из:\n"
    "- 'same' — повторение одного и того же факта.\n"
    "- 'supersedes_old' — новый факт явно отменяет/обновляет старый "
    "(новые данные, более точные значения, более поздние события).\n"
    "- 'contradiction' — противоречие, оба остаются с пометкой.\n"
    "- 'orthogonal' — на самом деле о разном (разные аспекты), оба остаются."
)


@dataclass(slots=True)
class StoreClaimsResult:
    affected_subjects: set[int] = field(default_factory=set)
    affected_claim_ids: list[int] = field(default_factory=list)


def _format_chunks(chunks: list[dict], summary: str | None = None) -> str:
    parts: list[str] = []
    if summary:
        parts.append("## Сводка документа\n" + summary.strip())
    for c in chunks:
        parts.append(f"## Chunk {c['ord']}\n{c['content']}")
    return "\n\n".join(parts)


def extract_claims(
    chunks: list[dict],
    needs_chunking: bool,
    summary: str | None,
    entity_canonical_names: list[str],
    llm: LLM,
) -> list[ExtractedClaim]:
    entity_list = "\n".join(f"- {name}" for name in entity_canonical_names) or "(пусто)"

    if not needs_chunking:
        text = (
            f"Известные сущности (используй ровно эти canonical_name):\n{entity_list}\n\n"
            + _format_chunks(chunks)
        )
        response = llm.structured(CLAIMS_SYSTEM, text, ClaimsResponse)
        for claim in response.claims:
            ord_set = {
                chunks[i]["ord"]
                for i in claim.citation_chunk_indices
                if 0 <= i < len(chunks)
            }
            claim.citation_chunk_indices = sorted(ord_set or {chunks[0]["ord"]})
        return response.claims

    aggregated: list[ExtractedClaim] = []
    for chunk in chunks:
        text = (
            f"Известные сущности (используй ровно эти canonical_name):\n{entity_list}\n\n"
            + _format_chunks([chunk], summary)
        )
        response = llm.structured(CLAIMS_SYSTEM, text, ClaimsResponse)
        for claim in response.claims:
            claim.citation_chunk_indices = [chunk["ord"]]
            aggregated.append(claim)
    return aggregated


def _arbitrate_supersession(
    new_claim: ExtractedClaim,
    candidate: dict,
    llm: LLM,
) -> SupersessionDecision:
    new_object = (
        new_claim.object_canonical_name
        if new_claim.object_kind == "entity"
        else new_claim.object_text
    )
    new_repr = (
        f"Subject: {new_claim.subject_canonical_name}\n"
        f"Predicate: {new_claim.predicate}\n"
        f"Object ({new_claim.object_kind}): {new_object}\n"
        f"Текст: {new_claim.claim_text}"
    )
    cand_object = (
        candidate.get("object_canonical_name")
        or candidate.get("object_text")
        or "—"
    )
    old_repr = (
        f"Predicate: {candidate['predicate']}\n"
        f"Object ({candidate['object_kind']}): {cand_object}\n"
        f"Текст: {candidate['claim_text']}\n"
        f"Уже подтверждено раз: {candidate.get('times_confirmed', 1)}"
    )
    user = (
        f"Новое утверждение:\n{new_repr}\n\n"
        f"Существующее утверждение:\n{old_repr}"
    )
    return llm.structured(SUPERSESSION_SYSTEM, user, SupersessionDecision)


def _resolve_object(
    extracted: ExtractedClaim,
    name_to_id: dict[str, int],
) -> tuple[str, int | None, str | None]:
    """Возвращает (object_kind, object_entity_id, object_text) для записи."""
    if extracted.object_kind == "entity" and extracted.object_canonical_name:
        eid = name_to_id.get(extracted.object_canonical_name)
        if eid is not None:
            return "entity", eid, None
        return "literal", None, extracted.object_canonical_name
    return "literal", None, (extracted.object_text or extracted.claim_text)


def _insert_with_canonical_predicate(
    direction_key: str,
    subject_id: int,
    predicate_text: str,
    object_kind: str,
    object_entity_id: int | None,
    object_text: str | None,
    claim_text: str,
    embedding: list[float],
    confidence_hint: float,
    embedder: Embedder,
    llm: LLM,
    claim_dao: ClaimDAO,
    predicate_dao: PredicateDAO,
    config: WikiConfig,
) -> int:
    canonical_predicate_id = normalize_predicate(
        direction_key=direction_key,
        predicate_text=predicate_text,
        embedder=embedder,
        llm=llm,
        predicate_dao=predicate_dao,
        config=config,
    )
    return claim_dao.insert(
        direction_key=direction_key,
        subject_entity_id=subject_id,
        predicate=predicate_text,
        object_kind=object_kind,
        object_entity_id=object_entity_id,
        object_text=object_text,
        claim_text=claim_text,
        claim_embedding=embedding,
        confidence=confidence_hint,
        canonical_predicate_id=canonical_predicate_id,
    )


def store_claims(
    direction_key: str,
    extracted_claims: list[ExtractedClaim],
    name_to_id: dict[str, int],
    chunk_ord_to_id: dict[int, int],
    embedder: Embedder,
    llm: LLM,
    claim_dao: ClaimDAO,
    predicate_dao: PredicateDAO,
    config: WikiConfig,
) -> StoreClaimsResult:
    """Записывает claims с supersession-логикой и predicate normalization.

    Возвращает (affected_subjects, affected_claim_ids):
    - affected_subjects: subject_entity_id-ы, чьи claims изменились → STEP 7.
    - affected_claim_ids: id-шники самих новых/обновлённых claim-ов → лог.
    """
    result = StoreClaimsResult()

    for ec in extracted_claims:
        subject_id = name_to_id.get(ec.subject_canonical_name)
        if subject_id is None:
            continue

        object_kind, object_entity_id, object_text = _resolve_object(ec, name_to_id)
        [emb] = embedder.embed([ec.claim_text])

        citation_chunk_ids = [
            chunk_ord_to_id[o]
            for o in ec.citation_chunk_indices
            if o in chunk_ord_to_id
        ]

        candidates = claim_dao.find_similar_for_subject(
            direction_key=direction_key,
            subject_entity_id=subject_id,
            predicate=ec.predicate,
            claim_embedding=emb,
            top_k=config.claim_supersession_pool,
            threshold=config.claim_supersession_threshold,
        )

        if not candidates:
            new_id = _insert_with_canonical_predicate(
                direction_key=direction_key,
                subject_id=subject_id,
                predicate_text=ec.predicate,
                object_kind=object_kind,
                object_entity_id=object_entity_id,
                object_text=object_text,
                claim_text=ec.claim_text,
                embedding=emb,
                confidence_hint=ec.confidence_hint,
                embedder=embedder,
                llm=llm,
                claim_dao=claim_dao,
                predicate_dao=predicate_dao,
                config=config,
            )
            claim_dao.add_citations(direction_key, new_id, citation_chunk_ids)
            result.affected_subjects.add(subject_id)
            result.affected_claim_ids.append(new_id)
            continue

        decision = _arbitrate_supersession(ec, candidates[0], llm)
        cand_id = candidates[0]["id"]

        if decision.decision == "same":
            claim_dao.confirm(
                cand_id,
                citation_chunk_ids,
                direction_key,
                hint_confidence=ec.confidence_hint,
            )
            result.affected_subjects.add(subject_id)
            result.affected_claim_ids.append(cand_id)
            continue

        new_id = _insert_with_canonical_predicate(
            direction_key=direction_key,
            subject_id=subject_id,
            predicate_text=ec.predicate,
            object_kind=object_kind,
            object_entity_id=object_entity_id,
            object_text=object_text,
            claim_text=ec.claim_text,
            embedding=emb,
            confidence_hint=ec.confidence_hint,
            embedder=embedder,
            llm=llm,
            claim_dao=claim_dao,
            predicate_dao=predicate_dao,
            config=config,
        )
        claim_dao.add_citations(direction_key, new_id, citation_chunk_ids)
        result.affected_claim_ids.append(new_id)

        if decision.decision == "supersedes_old":
            claim_dao.supersede(direction_key, cand_id, new_id, decision.reason)
        elif decision.decision == "contradiction":
            contradiction_id = claim_dao.add_contradiction(
                direction_key, cand_id, new_id, decision.reason
            )
            # Загружаем актуальное состояние обоих claim-ов для арбитра.
            claim_a = claim_dao.get(cand_id)
            claim_b = claim_dao.get(new_id)
            if claim_a and claim_b:
                resolve_contradiction(
                    direction_key=direction_key,
                    contradiction_id=contradiction_id,
                    claim_a=claim_a,
                    claim_b=claim_b,
                    llm=llm,
                    claim_dao=claim_dao,
                    config=config,
                )

        result.affected_subjects.add(subject_id)

    return result
