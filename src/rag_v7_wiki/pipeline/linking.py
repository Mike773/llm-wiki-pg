from __future__ import annotations

import re

from rag_v7_wiki.dao.entities import EntityDAO
from rag_v7_wiki.dao.pages import PageDAO


WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]\n|]+?)(?:\|([^\[\]\n]+?))?\]\]")


def _extract_wikilinks(content_md: str) -> list[tuple[str, str]]:
    """[(target_name, anchor_text)]. anchor_text может равняться target_name."""
    out: list[tuple[str, str]] = []
    for match in WIKILINK_PATTERN.finditer(content_md):
        target = match.group(1).strip()
        alias = (match.group(2) or target).strip()
        if target:
            out.append((target, alias))
    return out


def relink_pages(
    direction_key: str,
    page_ids: list[int],
    page_dao: PageDAO,
    entity_dao: EntityDAO,
) -> None:
    if not page_ids:
        return

    pages = page_dao.list_by_ids(direction_key, page_ids)
    for page in pages:
        page_id = page["id"]
        page_dao.clear_links(page_id)
        seen: set[str] = set()
        for target, anchor in _extract_wikilinks(page["content_md"] or ""):
            key = anchor.lower()
            if key in seen:
                continue
            seen.add(key)
            target_entity_id = entity_dao.find_by_alias(direction_key, target)
            page_dao.upsert_link(
                direction_key=direction_key,
                from_page_id=page_id,
                anchor_text=anchor,
                to_entity_id=target_entity_id,
            )
