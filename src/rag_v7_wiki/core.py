from __future__ import annotations

from typing import Any

import structlog
from psycopg_pool import ConnectionPool

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.dao.chunks import ChunkDAO
from rag_v7_wiki.dao.claims import ClaimDAO
from rag_v7_wiki.dao.connection import ConnectionManager
from rag_v7_wiki.dao.documents import DocumentDAO
from rag_v7_wiki.dao.entities import EntityDAO
from rag_v7_wiki.dao.log import WikiLogDAO
from rag_v7_wiki.dao.pages import PageDAO
from rag_v7_wiki.dao.predicates import PredicateDAO
from rag_v7_wiki.pipeline.chunking import chunk_document, summarize_document
from rag_v7_wiki.pipeline.claims import extract_claims, store_claims
from rag_v7_wiki.pipeline.coverage import apply_coverage
from rag_v7_wiki.pipeline.entities import extract_entities, resolve_and_upsert
from rag_v7_wiki.pipeline.indexing import rebuild_index_page, rebuild_log_page
from rag_v7_wiki.pipeline.linking import relink_pages
from rag_v7_wiki.pipeline.log import append_ingest_event
from rag_v7_wiki.pipeline.redact import redact
from rag_v7_wiki.pipeline.sources import synthesize_source_page
from rag_v7_wiki.pipeline.synthesis import synthesize_pages
from rag_v7_wiki.protocols import LLM, Embedder


log = structlog.get_logger(__name__)


class PipelineError(Exception):
    def __init__(self, step: str, original: Exception):
        super().__init__(f"{step}: {original}")
        self.step = step
        self.original = original


class WikiCore:
    def __init__(
        self,
        db_dsn: str | ConnectionPool,
        embedder: Embedder,
        llm: LLM,
        config: WikiConfig | None = None,
    ):
        self.cm = ConnectionManager(db_dsn)
        self.embedder = embedder
        self.llm = llm
        self.config = config or WikiConfig()

        if embedder.dim != self.config.expected_embedding_dim:
            raise ValueError(
                f"Embedder.dim={embedder.dim} не совпадает с "
                f"WikiConfig.expected_embedding_dim={self.config.expected_embedding_dim}. "
                "Размерность зашита в миграцию (vector(N)) — должна совпадать."
            )

        self.documents = DocumentDAO(self.cm)
        self.chunks = ChunkDAO(self.cm)
        self.entities = EntityDAO(self.cm)
        self.claims = ClaimDAO(self.cm)
        self.pages = PageDAO(self.cm)
        self.predicates = PredicateDAO(self.cm)
        self.log = WikiLogDAO(self.cm)

    def close(self) -> None:
        self.cm.close()

    def __enter__(self) -> "WikiCore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def ensure_direction(
        self,
        key: str,
        name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.documents.ensure_direction(key, name, description, settings)

    def process_document(self, direction_key: str, document_id: int) -> None:
        doc = self.documents.get(direction_key, document_id)
        if doc is None:
            raise ValueError(
                f"Document {document_id} not found in direction {direction_key}"
            )
        if doc["status"] == "processed":
            log.info(
                "document.already_processed",
                direction=direction_key,
                doc_id=document_id,
            )
            return
        try:
            self._run_pipeline(direction_key, doc)
            self.documents.set_status(direction_key, document_id, "processed")
        except PipelineError as exc:
            log.exception(
                "document.pipeline_failed",
                direction=direction_key,
                doc_id=document_id,
                step=exc.step,
            )
            self.documents.set_status(
                direction_key,
                document_id,
                "failed",
                failed_step=exc.step,
                error=str(exc.original),
            )
            raise
        except Exception as exc:
            log.exception(
                "document.pipeline_failed_unknown",
                direction=direction_key,
                doc_id=document_id,
            )
            self.documents.set_status(
                direction_key,
                document_id,
                "failed",
                failed_step="unknown",
                error=str(exc),
            )
            raise

    def process_pending(self, direction_key: str, limit: int = 10) -> list[int]:
        ids = self.documents.list_pending_ids(direction_key, limit)
        processed: list[int] = []
        for doc_id in ids:
            try:
                self.process_document(direction_key, doc_id)
                processed.append(doc_id)
            except Exception:
                # already logged + status='failed' записан
                continue
        return processed

    # ------------------------------------------------------------------
    # Public read API: index / log / source-pages
    # ------------------------------------------------------------------

    def get_index_md(self, direction_key: str) -> str | None:
        page = self.pages.get_singleton(direction_key, "index")
        return page["content_md"] if page else None

    def get_log_md(self, direction_key: str) -> str | None:
        page = self.pages.get_singleton(direction_key, "log")
        return page["content_md"] if page else None

    def get_source_page(
        self, direction_key: str, document_id: int
    ) -> dict[str, Any] | None:
        return self.pages.get_source_page(direction_key, document_id)

    def list_log_entries(
        self, direction_key: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self.log.list_recent(direction_key, limit=limit)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, direction_key: str, doc: dict[str, Any]) -> None:
        doc_id = doc["id"]

        # STEP 0.5 — PII / secret redaction (in-place over doc.content).
        try:
            redacted, matches = redact(
                doc["content"], strict=self.config.pii_strict_mode
            )
            if matches:
                redactions_json = [m.to_dict() for m in matches]
                self.documents.set_redacted_content(
                    direction_key, doc_id, redacted, redactions_json
                )
                doc["content"] = redacted
                doc["redactions"] = redactions_json
            # else: doc["redactions"] уже подгружен из БД get().
        except Exception as exc:
            raise PipelineError("redact", exc) from exc

        content = doc["content"]
        needs_chunking = doc["needs_chunking"]

        # STEP 1 — Chunking
        try:
            chunks_data = chunk_document(
                content=content,
                needs_chunking=needs_chunking,
                chunk_size_chars=self.config.chunk_size_chars,
                chunk_overlap_chars=self.config.chunk_overlap_chars,
            )
            self.chunks.bulk_insert(direction_key, doc_id, chunks_data)
            chunk_records = self.chunks.for_document(direction_key, doc_id)
            chunk_ord_to_id = {c["ord"]: c["id"] for c in chunk_records}
            self.documents.set_status(direction_key, doc_id, "chunked")

            if needs_chunking and not doc.get("summary"):
                summary_text = summarize_document(content, self.llm)
                self.documents.set_summary(direction_key, doc_id, summary_text)
                doc["summary"] = summary_text
            summary = doc.get("summary")
        except Exception as exc:
            raise PipelineError("chunking", exc) from exc

        # STEP 3+4 — Entity extraction & resolution
        try:
            llm_chunks = [
                {"id": c["id"], "ord": c["ord"], "content": c["content"]}
                for c in chunk_records
            ]
            extracted_entities = extract_entities(
                chunks=llm_chunks,
                needs_chunking=needs_chunking,
                summary=summary,
                llm=self.llm,
            )
            name_to_id = resolve_and_upsert(
                direction_key=direction_key,
                extracted=extracted_entities,
                chunk_ord_to_id=chunk_ord_to_id,
                embedder=self.embedder,
                llm=self.llm,
                entity_dao=self.entities,
                config=self.config,
            )
            self.documents.set_status(direction_key, doc_id, "entities_extracted")
        except Exception as exc:
            raise PipelineError("entities", exc) from exc

        # STEP 5+6 — Claim extraction, supersession, predicate normalization,
        # contradiction auto-resolution.
        try:
            extracted_claims = extract_claims(
                chunks=llm_chunks,
                needs_chunking=needs_chunking,
                summary=summary,
                entity_canonical_names=list({n: None for n in name_to_id}.keys()),
                llm=self.llm,
            )
            claims_result = store_claims(
                direction_key=direction_key,
                extracted_claims=extracted_claims,
                name_to_id=name_to_id,
                chunk_ord_to_id=chunk_ord_to_id,
                embedder=self.embedder,
                llm=self.llm,
                claim_dao=self.claims,
                predicate_dao=self.predicates,
                config=self.config,
            )
            self.documents.set_status(direction_key, doc_id, "claims_extracted")
        except Exception as exc:
            raise PipelineError("claims", exc) from exc

        # STEP 6b — Tier promotion
        try:
            self.claims.promote_tiers(
                direction_key=direction_key,
                episodic_min_confirmations=self.config.tier_promotion_episodic_min_confirmations,
                semantic_min_confirmations=self.config.tier_promotion_semantic_min_confirmations,
                semantic_min_age_days=self.config.tier_promotion_semantic_min_age_days,
            )
        except Exception as exc:
            raise PipelineError("tier_promotion", exc) from exc

        # STEP 7a — Source page synthesis
        try:
            source_page_id = synthesize_source_page(
                direction_key=direction_key,
                document=doc,
                summary=summary,
                known_entity_names=list(name_to_id.keys()),
                redactions=doc.get("redactions") or [],
                embedder=self.embedder,
                llm=self.llm,
                page_dao=self.pages,
                llm_model_name=self.llm.model_name,
            )
        except Exception as exc:
            raise PipelineError("source_page", exc) from exc

        # STEP 7 — Entity-page synthesis with quality pass
        try:
            page_records = synthesize_pages(
                direction_key=direction_key,
                affected_entity_ids=claims_result.affected_subjects,
                embedder=self.embedder,
                llm=self.llm,
                entity_dao=self.entities,
                claim_dao=self.claims,
                page_dao=self.pages,
                config=self.config,
                llm_model_name=self.llm.model_name,
            )
            entity_page_ids = [r["page_id"] for r in page_records]
            self.documents.set_status(direction_key, doc_id, "synthesized")
        except Exception as exc:
            raise PipelineError("synthesis", exc) from exc

        # STEP 8 — Cross-linking (entity-pages + source-page)
        try:
            relink_pages(
                direction_key=direction_key,
                page_ids=entity_page_ids + [source_page_id],
                page_dao=self.pages,
                entity_dao=self.entities,
            )
            self.documents.set_status(direction_key, doc_id, "linked")
        except Exception as exc:
            raise PipelineError("linking", exc) from exc

        # STEP 8b — Coverage / provenance after linking is final
        try:
            for record in page_records:
                apply_coverage(
                    direction_key=direction_key,
                    page_id=record["page_id"],
                    entity_id=record["entity_id"],
                    quality_score=record["quality_score"],
                    claim_dao=self.claims,
                    page_dao=self.pages,
                )
        except Exception as exc:
            raise PipelineError("coverage", exc) from exc

        # STEP 9 — Index page
        try:
            rebuild_index_page(
                direction_key=direction_key,
                page_dao=self.pages,
                embedder=self.embedder,
            )
        except Exception as exc:
            raise PipelineError("index", exc) from exc

        # STEP 10 — Log entry + log page
        try:
            redactions_summary = self._format_redactions_summary(
                doc.get("redactions") or []
            )
            append_ingest_event(
                direction_key=direction_key,
                document=doc,
                title=self._make_log_title(doc),
                affected_pages=entity_page_ids + [source_page_id],
                affected_claims=claims_result.affected_claim_ids,
                redactions_summary=redactions_summary,
                log_dao=self.log,
            )
            rebuild_log_page(
                direction_key=direction_key,
                log_dao=self.log,
                page_dao=self.pages,
                embedder=self.embedder,
                config=self.config,
            )
        except Exception as exc:
            raise PipelineError("log_index", exc) from exc

    @staticmethod
    def _format_redactions_summary(redactions: list[dict[str, Any]]) -> str | None:
        if not redactions:
            return None
        kinds: dict[str, int] = {}
        for r in redactions:
            k = str(r.get("kind", "unknown"))
            kinds[k] = kinds.get(k, 0) + 1
        return ", ".join(f"{k}×{n}" for k, n in sorted(kinds.items()))

    @staticmethod
    def _make_log_title(doc: dict[str, Any]) -> str:
        ext = doc.get("external_id")
        if ext:
            return str(ext)
        snippet = (doc.get("content") or "").strip().splitlines()[0:1]
        first_line = snippet[0] if snippet else f"document #{doc['id']}"
        return first_line[:80]
