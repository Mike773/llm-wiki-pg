"""Синтез страницы-источника на ingest.

Каждый успешно обработанный документ получает свою wiki-страницу
(`page_kind='source'`), где LLM фиксирует: краткое содержание, ключевые
тезисы, упомянутые сущности (как `[[wikilinks]]`). Эта страница
обновляется при reprocess того же документа.
"""

from __future__ import annotations

import re
from typing import Any

from rag_v7_wiki.dao.pages import PageDAO
from rag_v7_wiki.protocols import LLM, Embedder
from rag_v7_wiki.schemas import SourcePageResponse


SOURCE_PAGE_SYSTEM = (
    "Ты составляешь карточку-источника для wiki. На вход — текст документа "
    "(возможно усечённый) и список уже распознанных канонических сущностей. "
    "Верни:\n"
    "- title: короткий заголовок источника, отражающий его предмет.\n"
    "- abstract: 3–5 предложений, что это за источник и о чём он.\n"
    "- key_takeaways: ключевые тезисы из источника (по одному на пункт).\n"
    "- top_entities: канонические имена сущностей источника, которые войдут "
    "в страницу как [[wikilinks]]. Бери ровно из списка известных."
)


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-а-яёіїєґ]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "source"


def _build_user_prompt(
    document_content: str,
    summary: str | None,
    known_entities: list[str],
) -> str:
    parts: list[str] = []
    if known_entities:
        parts.append(
            "Известные сущности (используй их каноническое имя для wikilinks):\n"
            + "\n".join(f"- {name}" for name in known_entities)
        )
    else:
        parts.append("Известные сущности: (пусто)")

    if summary:
        parts.append("\nСводка документа:\n" + summary.strip())

    # Ограничим контекст ~ 12k символов; для больших документов summary
    # содержит главное, остальное — иллюстративный отрывок.
    excerpt = document_content[:12000]
    if len(document_content) > len(excerpt):
        excerpt += "\n…(документ усечён)"
    parts.append("\nИсходный документ:\n" + excerpt)

    return "\n".join(parts)


def _render_markdown(
    response: SourcePageResponse,
    document: dict[str, Any],
    redactions: list[dict[str, Any]] | None,
) -> str:
    lines: list[str] = [f"# {response.title.strip() or 'Источник'}", ""]
    lines.append("> Страница-источник, автогенерируется при ingest.")
    lines.append("")

    meta_lines: list[str] = []
    ext = document.get("external_id")
    if ext:
        meta_lines.append(f"- **external_id**: `{ext}`")
    meta_lines.append(f"- **document_id**: `{document['id']}`")
    if document.get("status"):
        meta_lines.append(f"- **status**: `{document['status']}`")
    if redactions:
        kinds: dict[str, int] = {}
        for r in redactions:
            k = r.get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1
        red_str = ", ".join(f"{k}×{n}" for k, n in sorted(kinds.items()))
        meta_lines.append(f"- **redactions**: {red_str}")
    lines.extend(meta_lines)
    lines.append("")

    lines.append("## Аннотация")
    lines.append(response.abstract.strip() or "—")
    lines.append("")

    if response.key_takeaways:
        lines.append("## Ключевые тезисы")
        for t in response.key_takeaways:
            t_stripped = t.strip().rstrip(".")
            if t_stripped:
                lines.append(f"- {t_stripped}.")
        lines.append("")

    if response.top_entities:
        lines.append("## Упомянутые сущности")
        seen: set[str] = set()
        for ent in response.top_entities:
            ent = ent.strip()
            if ent and ent.lower() not in seen:
                seen.add(ent.lower())
                lines.append(f"- [[{ent}]]")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def synthesize_source_page(
    direction_key: str,
    document: dict[str, Any],
    summary: str | None,
    known_entity_names: list[str],
    redactions: list[dict[str, Any]] | None,
    embedder: Embedder,
    llm: LLM,
    page_dao: PageDAO,
    llm_model_name: str | None = None,
) -> int:
    """Создаёт/обновляет страницу-источника. Возвращает page_id."""
    user_prompt = _build_user_prompt(
        document_content=document["content"],
        summary=summary,
        known_entities=known_entity_names,
    )
    response = llm.structured(SOURCE_PAGE_SYSTEM, user_prompt, SourcePageResponse)

    title = response.title.strip() or f"Источник #{document['id']}"
    content_md = _render_markdown(response, document, redactions)
    [content_emb] = embedder.embed([content_md])

    slug = f"src-{document['id']}-{_slugify(title)}"
    page_id, version = page_dao.upsert_source_page(
        direction_key=direction_key,
        source_document_id=document["id"],
        slug=slug,
        title=title,
        content_md=content_md,
        content_embedding=content_emb,
    )
    page_dao.save_revision(
        page_id=page_id,
        version=version,
        content_md=content_md,
        synthesized_from_claim_ids=[],
        llm_model=llm_model_name,
    )
    return page_id
