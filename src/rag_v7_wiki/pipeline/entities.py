from __future__ import annotations

import json

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.entities import EntityDAO
from rag_v7_wiki.protocols import LLM, Embedder
from rag_v7_wiki.schemas import (
    EntitiesResponse,
    ExtractedEntity,
    ResolutionDecision,
)

EXTRACT_SYSTEM = (
    "Ты извлекаешь именованные сущности из текста. Возвращай только то, что "
    "действительно встречается в тексте. Для каждой сущности укажи:\n"
    "- entity_type: одно из Person, Project, Library, Tool, Concept, Decision, "
    "Event, Organization, Location, Other.\n"
    "- canonical_name: лучшее каноническое имя.\n"
    "- aliases: другие написания/синонимы, если упомянуты.\n"
    "- salient_attrs: ключ-значение из текста (опционально).\n"
    "- supporting_chunk_indices: индексы чанков (по нумерации '## Chunk N'), "
    "где сущность упомянута. Если показан только один чанк — указывай [0]."
)

RESOLVE_SYSTEM = (
    "Ты решаешь, описывает ли новая извлечённая сущность ту же сущность, что "
    "и один из ранее известных кандидатов. Учитывай тип, имя, синонимы и "
    "атрибуты. Если сомневаешься — отвечай 'new'."
)


def _format_chunks(chunks: list[dict], summary: str | None = None) -> str:
    parts: list[str] = []
    if summary:
        parts.append("## Сводка документа\n" + summary.strip())
    for c in chunks:
        parts.append(f"## Chunk {c['ord']}\n{c['content']}")
    return "\n\n".join(parts)


def extract_entities(
    chunks: list[dict],
    needs_chunking: bool,
    summary: str | None,
    llm: LLM,
) -> list[ExtractedEntity]:
    """Извлекает сущности.

    Если needs_chunking=False, делает один LLM-вызов на весь документ.
    Иначе — per chunk с summary в контексте; результаты дедуплицируются по
    (entity_type, canonical_name).
    """
    if not needs_chunking:
        text = _format_chunks(chunks)
        response = llm.structured(EXTRACT_SYSTEM, text, EntitiesResponse)
        for e in response.entities:
            ord_set = {chunks[i]["ord"] for i in e.supporting_chunk_indices if 0 <= i < len(chunks)}
            if not ord_set:
                ord_set = {chunks[0]["ord"]}
            e.supporting_chunk_indices = sorted(ord_set)
        return response.entities

    aggregated: dict[tuple[str, str], ExtractedEntity] = {}
    for chunk in chunks:
        text = _format_chunks([chunk], summary)
        response = llm.structured(EXTRACT_SYSTEM, text, EntitiesResponse)
        for e in response.entities:
            key = (e.entity_type, e.canonical_name)
            if key in aggregated:
                ag = aggregated[key]
                ag.aliases = sorted({*ag.aliases, *e.aliases})
                ag.salient_attrs = {**ag.salient_attrs, **e.salient_attrs}
                ag.supporting_chunk_indices = sorted(
                    {*ag.supporting_chunk_indices, chunk["ord"]}
                )
            else:
                e.supporting_chunk_indices = [chunk["ord"]]
                aggregated[key] = e
    return list(aggregated.values())


def _arbitrate_resolution(
    new_entity: ExtractedEntity,
    candidates: list[dict],
    llm: LLM,
) -> ResolutionDecision:
    new_repr_lines = [
        f"Тип: {new_entity.entity_type}",
        f"Каноническое имя: {new_entity.canonical_name}",
    ]
    if new_entity.aliases:
        new_repr_lines.append("Алиасы: " + ", ".join(new_entity.aliases))
    if new_entity.salient_attrs:
        new_repr_lines.append(
            "Атрибуты: " + json.dumps(new_entity.salient_attrs, ensure_ascii=False)
        )
    new_repr = "\n".join(new_repr_lines)

    cand_lines = []
    for i, c in enumerate(candidates):
        attrs = c.get("salient_attrs") or {}
        cand_lines.append(
            f"[{i}] {c['canonical_name']} (sim={c.get('similarity', 0):.2f}) "
            f"attrs={json.dumps(attrs, ensure_ascii=False)}"
        )

    user = (
        f"Новая сущность:\n{new_repr}\n\n"
        f"Кандидаты (та же сущность?):\n" + "\n".join(cand_lines)
    )
    return llm.structured(RESOLVE_SYSTEM, user, ResolutionDecision)


def resolve_and_upsert(
    direction_key: str,
    extracted: list[ExtractedEntity],
    chunk_ord_to_id: dict[int, int],
    embedder: Embedder,
    llm: LLM,
    entity_dao: EntityDAO,
    config: WikiConfig,
) -> dict[str, int]:
    """Возвращает mapping canonical_name (и алиасов) → entity_id.

    Используется на следующем шаге (claim extraction) чтобы привязать
    subject/object к существующим entity-идентификаторам.
    """
    name_to_id: dict[str, int] = {}

    for entity in extracted:
        if not entity.canonical_name.strip():
            continue
        [embedding] = embedder.embed([entity.canonical_name])

        candidates = entity_dao.find_similar(
            direction_key,
            entity.entity_type,
            embedding,
            top_k=config.entity_resolution_pool,
            threshold=config.entity_resolution_threshold,
        )

        entity_id: int | None = None
        if candidates:
            decision = _arbitrate_resolution(entity, candidates, llm)
            if (
                decision.decision == "same"
                and decision.matched_candidate_index is not None
                and 0 <= decision.matched_candidate_index < len(candidates)
            ):
                entity_id = candidates[decision.matched_candidate_index]["id"]
                if entity.salient_attrs:
                    entity_dao.merge_attrs(entity_id, entity.salient_attrs)

        if entity_id is None:
            entity_id = entity_dao.upsert(
                direction_key=direction_key,
                entity_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                canonical_name_embedding=embedding,
                salient_attrs=entity.salient_attrs or None,
            )

        for alias in entity.aliases:
            if alias and alias != entity.canonical_name:
                entity_dao.add_alias(direction_key, entity_id, alias)

        added_mentions = 0
        for ord_ in entity.supporting_chunk_indices:
            chunk_id = chunk_ord_to_id.get(ord_)
            if chunk_id is not None:
                if entity_dao.add_mention(
                    direction_key, entity_id, chunk_id, entity.canonical_name
                ):
                    added_mentions += 1
        if added_mentions:
            entity_dao.bump_mention_count(entity_id, by=added_mentions)

        name_to_id[entity.canonical_name] = entity_id
        for alias in entity.aliases:
            name_to_id.setdefault(alias, entity_id)

    return name_to_id
