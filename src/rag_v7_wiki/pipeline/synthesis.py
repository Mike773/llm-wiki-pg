from __future__ import annotations

import json
import re

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.claims import ClaimDAO
from rag_v7_wiki.dao.entities import EntityDAO
from rag_v7_wiki.dao.pages import PageDAO
from rag_v7_wiki.pipeline.quality import build_facts_block, quality_pass
from rag_v7_wiki.protocols import LLM, Embedder
from rag_v7_wiki.schemas import PageSynthesisResponse


SYNTHESIS_SYSTEM = (
    "Ты редактируешь wiki-страницу о сущности на основе перечисленных фактов. "
    "Пиши markdown: заголовок (H1), короткое определение, секция «Свойства» с "
    "пунктами, секция «Связи» с пунктами и [[wikilink]]ами на канонические имена "
    "других сущностей. Не добавляй ничего сверх фактов и существующего контекста. "
    "Если страница уже существует — мерджи новые факты в её структуру, не теряя "
    "корректную информацию."
)


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-а-яёіїєґ]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "page"


def _build_user_prompt(
    entity: dict,
    claims: list[dict],
    existing_page: dict | None,
    quality_feedback: str | None = None,
) -> str:
    fact_lines = []
    for c in claims:
        obj = (
            c.get("object_canonical_name")
            or c.get("object_text")
            or "—"
        )
        fact_lines.append(
            f"- [{c['predicate']}] → {obj} :: {c['claim_text']} "
            f"(подтверждено {c['times_confirmed']}×, conf={c['confidence']:.2f})"
        )
    facts_block = "\n".join(fact_lines) if fact_lines else "(нет фактов)"

    parts = [
        f"Сущность: {entity['canonical_name']}",
        f"Тип: {entity['entity_type']}",
        f"Атрибуты: {json.dumps(entity.get('salient_attrs') or {}, ensure_ascii=False)}",
        "",
        "Факты:",
        facts_block,
    ]

    if existing_page and existing_page.get("content_md"):
        parts += [
            "",
            "Текущая версия страницы (обнови, не теряй полезное):",
            "```markdown",
            existing_page["content_md"],
            "```",
        ]
    else:
        parts += ["", "Это новая страница. Сгенерируй с нуля."]

    if quality_feedback:
        parts += [
            "",
            "Замечания по качеству к предыдущей версии (учти при пересборке):",
            quality_feedback,
        ]

    return "\n".join(parts)


def synthesize_pages(
    direction_key: str,
    affected_entity_ids: set[int],
    embedder: Embedder,
    llm: LLM,
    entity_dao: EntityDAO,
    claim_dao: ClaimDAO,
    page_dao: PageDAO,
    config: WikiConfig,
    llm_model_name: str | None = None,
) -> list[dict]:
    """Синтезирует/обновляет wiki-страницы для затронутых entities.

    После основного синтеза прогоняет quality-pass (с возможным re-synth).
    Coverage-метрики и provenance применяются позже (в core, после relink).

    Возвращает list of dict с ключами: page_id, entity_id, quality_score.
    """
    affected_pages: list[dict] = []

    for entity_id in affected_entity_ids:
        entity = entity_dao.get(direction_key, entity_id)
        if not entity:
            continue

        claims = claim_dao.claims_for_entity(
            direction_key, entity_id, only_active=True
        )
        if not claims:
            continue

        existing_page = page_dao.get_by_entity(direction_key, entity_id)
        user_prompt = _build_user_prompt(entity, claims, existing_page)
        response = llm.structured(
            SYNTHESIS_SYSTEM, user_prompt, PageSynthesisResponse
        )
        title = response.title.strip() or entity["canonical_name"]
        content_md = response.content_md.strip()

        def _resynthesize(
            feedback: str,
            previous_content: str,
            _entity: dict = entity,
            _claims: list[dict] = claims,
        ) -> tuple[str, str]:
            prompt = _build_user_prompt(
                _entity,
                _claims,
                {"content_md": previous_content},
                quality_feedback=feedback,
            )
            re_response = llm.structured(
                SYNTHESIS_SYSTEM, prompt, PageSynthesisResponse
            )
            return (
                re_response.title.strip() or _entity["canonical_name"],
                re_response.content_md.strip(),
            )

        facts_block = build_facts_block(claims)
        title, content_md, quality = quality_pass(
            title=title,
            content_md=content_md,
            facts_block=facts_block,
            llm=llm,
            config=config,
            resynthesize=_resynthesize,
        )

        [content_emb] = embedder.embed([content_md])

        slug = f"{_slugify(title)}-{entity_id}"
        page_id, version = page_dao.upsert(
            direction_key=direction_key,
            entity_id=entity_id,
            slug=slug,
            title=title,
            content_md=content_md,
            content_embedding=content_emb,
        )
        page_dao.save_revision(
            page_id=page_id,
            version=version,
            content_md=content_md,
            synthesized_from_claim_ids=[c["id"] for c in claims],
            llm_model=llm_model_name,
            quality_score=quality.score,
        )

        affected_pages.append(
            {
                "page_id": page_id,
                "entity_id": entity_id,
                "quality_score": quality.score,
            }
        )

    return affected_pages
