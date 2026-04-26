"""Coverage-метрики и provenance для wiki-страниц.

После synthesize_pages для каждой entity-страницы считаем:
- coverage_claims: число активных claim-ов по subject_entity_id;
- coverage_unresolved_links: число page_links с resolved=false;
- coverage_contradictions: число `flagged_contradiction` claim-ов сущности.

И записываем `page_sources(page_id, document_id, claim_count)` — прямую
связь между страницей и документами, из которых она построена.
"""

from __future__ import annotations

from rag_v7_wiki.dao.claims import ClaimDAO
from rag_v7_wiki.dao.pages import PageDAO


def apply_coverage(
    direction_key: str,
    page_id: int,
    entity_id: int,
    quality_score: float | None,
    claim_dao: ClaimDAO,
    page_dao: PageDAO,
) -> None:
    active_claims = claim_dao.claims_for_entity(direction_key, entity_id, only_active=True)
    coverage_claims = len(active_claims)
    coverage_contradictions = claim_dao.count_contradictions_for_entity(
        direction_key, entity_id
    )
    coverage_unresolved = page_dao.count_unresolved_links(page_id)

    page_dao.set_metrics(
        page_id=page_id,
        quality_score=quality_score,
        coverage_claims=coverage_claims,
        coverage_unresolved_links=coverage_unresolved,
        coverage_contradictions=coverage_contradictions,
        body_meta=None,
    )

    document_counts = claim_dao.documents_for_entity(direction_key, entity_id)
    page_dao.upsert_provenance(direction_key, page_id, document_counts)
