"""Пересборка singleton-страниц index.md и log.md в Postgres.

Источники правды:
- index.md рендерится из rag_v7.wiki_pages (все страницы кроме самих index/log).
- log.md рендерится из rag_v7.wiki_log_entries (последние N записей).

Эти страницы — singleton per direction (`page_kind ∈ {'index','log'}`),
обновляются на каждом ingest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.log import WikiLogDAO
from rag_v7_wiki.dao.pages import PageDAO
from rag_v7_wiki.protocols import Embedder


_KIND_HEADERS: dict[str, str] = {
    "entity": "Сущности",
    "source": "Источники",
    "concept": "Концепции",
    "comparison": "Сравнения",
    "overview": "Обзоры",
}


def _fmt_ts(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(value or "")


def _render_index_md(direction_key: str, pages: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Index — {direction_key}",
        "",
        f"_Сгенерировано: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Всего страниц: {len(pages)}_",
        "",
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        grouped.setdefault(p["page_kind"], []).append(p)

    for kind in ("entity", "source", "concept", "comparison", "overview"):
        bucket = grouped.get(kind)
        if not bucket:
            continue
        lines.append(f"## {_KIND_HEADERS.get(kind, kind.title())} ({len(bucket)})")
        lines.append("")
        for p in bucket:
            link_target = p["title"]
            quality = (
                f" · q={p['quality_score']:.2f}"
                if p.get("quality_score") is not None
                else ""
            )
            extras: list[str] = []
            if p.get("entity_type"):
                extras.append(p["entity_type"])
            if p.get("coverage_claims") is not None:
                extras.append(f"claims={p['coverage_claims']}")
            if p.get("coverage_contradictions"):
                extras.append(f"⚠{p['coverage_contradictions']}")
            extras_str = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"- [[{link_target}]]{extras_str}{quality}")
        lines.append("")

    other_kinds = set(grouped) - {"entity", "source", "concept", "comparison", "overview", "index", "log"}
    for kind in sorted(other_kinds):
        bucket = grouped[kind]
        lines.append(f"## {kind} ({len(bucket)})")
        lines.append("")
        for p in bucket:
            lines.append(f"- [[{p['title']}]]")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_log_md(direction_key: str, entries: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Log — {direction_key}",
        "",
        f"_Записей: {len(entries)}_",
        "",
    ]

    for e in entries:
        ts = _fmt_ts(e.get("ts"))
        date = ts.split(" ")[0] if ts else ""
        lines.append(f"## [{date}] {e['event_kind']} | {e['title']}")
        if e.get("summary"):
            lines.append(e["summary"])
        if e.get("affected_pages"):
            n = len(e["affected_pages"])
            lines.append(f"_pages: {n}_")
        if e.get("affected_claims"):
            n = len(e["affected_claims"])
            lines.append(f"_claims: {n}_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def rebuild_index_page(
    direction_key: str,
    page_dao: PageDAO,
    embedder: Embedder,
) -> int:
    """Пересобирает singleton-страницу `page_kind='index'`. Возвращает page_id."""
    pages = page_dao.list_for_index(direction_key)
    content_md = _render_index_md(direction_key, pages)
    [emb] = embedder.embed([content_md])
    page_id, version = page_dao.upsert_singleton_page(
        direction_key=direction_key,
        page_kind="index",
        slug="index",
        title=f"Index — {direction_key}",
        content_md=content_md,
        content_embedding=emb,
    )
    page_dao.save_revision(
        page_id=page_id,
        version=version,
        content_md=content_md,
        synthesized_from_claim_ids=[],
        llm_model=None,
    )
    return page_id


def rebuild_log_page(
    direction_key: str,
    log_dao: WikiLogDAO,
    page_dao: PageDAO,
    embedder: Embedder,
    config: WikiConfig,
) -> int:
    """Пересобирает singleton-страницу `page_kind='log'`. Возвращает page_id."""
    entries = log_dao.list_recent(direction_key, limit=config.log_page_recent_entries)
    content_md = _render_log_md(direction_key, entries)
    [emb] = embedder.embed([content_md])
    page_id, version = page_dao.upsert_singleton_page(
        direction_key=direction_key,
        page_kind="log",
        slug="log",
        title=f"Log — {direction_key}",
        content_md=content_md,
        content_embedding=emb,
    )
    page_dao.save_revision(
        page_id=page_id,
        version=version,
        content_md=content_md,
        synthesized_from_claim_ids=[],
        llm_model=None,
    )
    return page_id
