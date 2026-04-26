"""Append-only лог операций над wiki.

Карпатый описывает `log.md` с форматом `## [YYYY-MM-DD] ingest | <title>`.
В нашей реализации источник правды — таблица `wiki_log_entries`. Файла на
диске нет, но `pipeline/indexing.py` рендерит из этих строк markdown
страницы `page_kind='log'`.
"""

from __future__ import annotations

from typing import Any

from rag_v7_wiki.dao.log import WikiLogDAO


def append_ingest_event(
    direction_key: str,
    document: dict[str, Any],
    title: str,
    affected_pages: list[int],
    affected_claims: list[int],
    redactions_summary: str | None,
    log_dao: WikiLogDAO,
) -> int:
    """Записывает строку лога об успешном ingest-е документа."""
    summary_parts: list[str] = []
    if document.get("external_id"):
        summary_parts.append(f"external_id={document['external_id']}")
    summary_parts.append(f"pages={len(affected_pages)}")
    summary_parts.append(f"claims={len(affected_claims)}")
    if redactions_summary:
        summary_parts.append(f"redactions=({redactions_summary})")
    summary = "; ".join(summary_parts)

    return log_dao.append(
        direction_key=direction_key,
        event_kind="ingest",
        title=title,
        ref_document_id=document["id"],
        summary=summary,
        affected_pages=affected_pages,
        affected_claims=affected_claims,
    )
