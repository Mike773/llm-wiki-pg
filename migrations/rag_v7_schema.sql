-- ============================================================
--  rag_v7_wiki schema (single-file, idempotent)
--  LLM Wiki v2: documents -> chunks -> entities -> claims -> wiki_pages
--  All vector columns are vector(2560).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS rag_v7;

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

-- ------------------------------------------------------------
--  ENUM types
-- ------------------------------------------------------------

DO $$ BEGIN
    CREATE TYPE rag_v7.document_status AS ENUM (
        'pending',
        'chunked',
        'embedded',
        'entities_extracted',
        'claims_extracted',
        'synthesized',
        'linked',
        'processed',
        'failed'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.claim_status AS ENUM (
        'active',
        'superseded',
        'flagged_contradiction'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.claim_object_kind AS ENUM (
        'entity',
        'literal'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.claim_tier AS ENUM (
        'working',
        'episodic',
        'semantic',
        'procedural'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.contradiction_status AS ENUM (
        'open',
        'resolved',
        'accepted_both'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.wiki_page_kind AS ENUM (
        'entity',
        'source',
        'concept',
        'comparison',
        'overview',
        'index',
        'log'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE rag_v7.log_event_kind AS ENUM (
        'ingest',
        'reprocess',
        'lint',
        'query',
        'manual'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

-- ------------------------------------------------------------
--  Tables
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS rag_v7.directions (
    key          TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    settings     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_v7.documents (
    id                 BIGSERIAL PRIMARY KEY,
    direction_key      TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    external_id        TEXT,
    content            TEXT NOT NULL,
    needs_chunking     BOOLEAN NOT NULL DEFAULT false,
    status             rag_v7.document_status NOT NULL DEFAULT 'pending',
    failed_step        TEXT,
    error              TEXT,
    summary            TEXT,
    summary_embedding  public.vector(2560),
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    redactions         JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at       TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_direction_external_uniq
    ON rag_v7.documents (direction_key, external_id)
    WHERE external_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_direction_status_idx
    ON rag_v7.documents (direction_key, status);


CREATE TABLE IF NOT EXISTS rag_v7.chunks (
    id             BIGSERIAL PRIMARY KEY,
    direction_key  TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    document_id    BIGINT NOT NULL REFERENCES rag_v7.documents(id) ON DELETE CASCADE,
    ord            INT NOT NULL,
    content        TEXT NOT NULL,
    length         INT NOT NULL,                            -- длина в символах
    embedding      public.vector(2560),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_direction_idx
    ON rag_v7.chunks (direction_key);

CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON rag_v7.chunks (document_id);


CREATE TABLE IF NOT EXISTS rag_v7.entities (
    id                        BIGSERIAL PRIMARY KEY,
    direction_key             TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    entity_type               TEXT NOT NULL,
    canonical_name            TEXT NOT NULL,
    canonical_name_embedding  public.vector(2560) NOT NULL,
    salient_attrs             JSONB NOT NULL DEFAULT '{}'::jsonb,
    mention_count             INT NOT NULL DEFAULT 0,
    confidence                REAL NOT NULL DEFAULT 0.5,
    first_seen_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (direction_key, entity_type, canonical_name)
);

CREATE INDEX IF NOT EXISTS entities_direction_type_idx
    ON rag_v7.entities (direction_key, entity_type);


CREATE TABLE IF NOT EXISTS rag_v7.entity_aliases (
    id              BIGSERIAL PRIMARY KEY,
    entity_id       BIGINT NOT NULL REFERENCES rag_v7.entities(id) ON DELETE CASCADE,
    direction_key   TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    alias           TEXT NOT NULL,
    alias_embedding public.vector(2560),
    source          TEXT NOT NULL DEFAULT 'extracted',
    UNIQUE (entity_id, alias)
);

CREATE INDEX IF NOT EXISTS entity_aliases_direction_idx
    ON rag_v7.entity_aliases (direction_key);


CREATE TABLE IF NOT EXISTS rag_v7.entity_mentions (
    id              BIGSERIAL PRIMARY KEY,
    entity_id       BIGINT NOT NULL REFERENCES rag_v7.entities(id) ON DELETE CASCADE,
    chunk_id        BIGINT NOT NULL REFERENCES rag_v7.chunks(id) ON DELETE CASCADE,
    direction_key   TEXT NOT NULL,
    extracted_form  TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS entity_mentions_chunk_idx
    ON rag_v7.entity_mentions (chunk_id);


CREATE TABLE IF NOT EXISTS rag_v7.canonical_predicates (
    id            BIGSERIAL PRIMARY KEY,
    direction_key TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    canonical     TEXT NOT NULL,
    embedding     public.vector(2560) NOT NULL,
    description   TEXT,
    times_used    INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (direction_key, canonical)
);

CREATE INDEX IF NOT EXISTS canonical_predicates_direction_idx
    ON rag_v7.canonical_predicates (direction_key);


CREATE TABLE IF NOT EXISTS rag_v7.claims (
    id                       BIGSERIAL PRIMARY KEY,
    direction_key            TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    subject_entity_id        BIGINT NOT NULL REFERENCES rag_v7.entities(id) ON DELETE CASCADE,
    predicate                TEXT NOT NULL,
    canonical_predicate_id   BIGINT REFERENCES rag_v7.canonical_predicates(id) ON DELETE SET NULL,
    object_kind              rag_v7.claim_object_kind NOT NULL,
    object_entity_id         BIGINT REFERENCES rag_v7.entities(id) ON DELETE SET NULL,
    object_text              TEXT,
    claim_text               TEXT NOT NULL,
    claim_embedding          public.vector(2560) NOT NULL,
    confidence               REAL NOT NULL DEFAULT 0.5,
    times_confirmed          INT NOT NULL DEFAULT 1,
    tier                     rag_v7.claim_tier NOT NULL DEFAULT 'working',
    first_seen_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    status                   rag_v7.claim_status NOT NULL DEFAULT 'active',
    superseded_by_id         BIGINT REFERENCES rag_v7.claims(id) ON DELETE SET NULL,
    CONSTRAINT claims_object_consistency CHECK (
        (object_kind = 'entity'  AND object_entity_id IS NOT NULL AND object_text IS NULL)
        OR
        (object_kind = 'literal' AND object_text      IS NOT NULL AND object_entity_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS claims_direction_subject_predicate_idx
    ON rag_v7.claims (direction_key, subject_entity_id, predicate);

CREATE INDEX IF NOT EXISTS claims_direction_status_idx
    ON rag_v7.claims (direction_key, status);

CREATE INDEX IF NOT EXISTS claims_object_entity_idx
    ON rag_v7.claims (object_entity_id)
    WHERE object_entity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS claims_canonical_predicate_idx
    ON rag_v7.claims (direction_key, subject_entity_id, canonical_predicate_id)
    WHERE canonical_predicate_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS claims_tier_idx
    ON rag_v7.claims (direction_key, tier);


CREATE TABLE IF NOT EXISTS rag_v7.claim_citations (
    claim_id      BIGINT NOT NULL REFERENCES rag_v7.claims(id) ON DELETE CASCADE,
    chunk_id      BIGINT NOT NULL REFERENCES rag_v7.chunks(id) ON DELETE CASCADE,
    direction_key TEXT NOT NULL,
    PRIMARY KEY (claim_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS claim_citations_chunk_idx
    ON rag_v7.claim_citations (chunk_id);


CREATE TABLE IF NOT EXISTS rag_v7.claim_supersedes (
    id            BIGSERIAL PRIMARY KEY,
    old_claim_id  BIGINT NOT NULL REFERENCES rag_v7.claims(id) ON DELETE CASCADE,
    new_claim_id  BIGINT NOT NULL REFERENCES rag_v7.claims(id) ON DELETE CASCADE,
    direction_key TEXT NOT NULL,
    reason        TEXT NOT NULL,
    decided_by    TEXT NOT NULL DEFAULT 'llm_arbiter',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (old_claim_id, new_claim_id)
);


CREATE TABLE IF NOT EXISTS rag_v7.claim_contradictions (
    id            BIGSERIAL PRIMARY KEY,
    direction_key TEXT NOT NULL,
    claim_a_id    BIGINT NOT NULL REFERENCES rag_v7.claims(id) ON DELETE CASCADE,
    claim_b_id    BIGINT NOT NULL REFERENCES rag_v7.claims(id) ON DELETE CASCADE,
    notes         TEXT,
    status        rag_v7.contradiction_status NOT NULL DEFAULT 'open',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at   TIMESTAMPTZ
);


CREATE TABLE IF NOT EXISTS rag_v7.wiki_pages (
    id                          BIGSERIAL PRIMARY KEY,
    direction_key               TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    entity_id                   BIGINT REFERENCES rag_v7.entities(id) ON DELETE CASCADE,
    page_kind                   rag_v7.wiki_page_kind NOT NULL DEFAULT 'entity',
    source_document_id          BIGINT REFERENCES rag_v7.documents(id) ON DELETE CASCADE,
    slug                        TEXT NOT NULL,
    title                       TEXT NOT NULL,
    content_md                  TEXT NOT NULL,
    content_embedding           public.vector(2560) NOT NULL,
    version                     INT NOT NULL DEFAULT 1,
    last_synthesized_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    quality_score               REAL,
    coverage_claims             INT NOT NULL DEFAULT 0,
    coverage_unresolved_links   INT NOT NULL DEFAULT 0,
    coverage_contradictions     INT NOT NULL DEFAULT 0,
    body_meta                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (direction_key, slug)
);

CREATE UNIQUE INDEX IF NOT EXISTS wiki_pages_dir_entity_uniq
    ON rag_v7.wiki_pages (direction_key, entity_id)
    WHERE entity_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS wiki_pages_dir_source_uniq
    ON rag_v7.wiki_pages (direction_key, source_document_id)
    WHERE source_document_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS wiki_pages_dir_singleton_uniq
    ON rag_v7.wiki_pages (direction_key, page_kind)
    WHERE page_kind IN ('index', 'log', 'overview');

CREATE INDEX IF NOT EXISTS wiki_pages_direction_kind_idx
    ON rag_v7.wiki_pages (direction_key, page_kind);


CREATE TABLE IF NOT EXISTS rag_v7.wiki_page_revisions (
    id                          BIGSERIAL PRIMARY KEY,
    page_id                     BIGINT NOT NULL REFERENCES rag_v7.wiki_pages(id) ON DELETE CASCADE,
    version                     INT NOT NULL,
    content_md                  TEXT NOT NULL,
    synthesized_from_claim_ids  BIGINT[] NOT NULL,
    llm_model                   TEXT,
    quality_score               REAL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, version)
);


CREATE TABLE IF NOT EXISTS rag_v7.page_links (
    id            BIGSERIAL PRIMARY KEY,
    direction_key TEXT NOT NULL,
    from_page_id  BIGINT NOT NULL REFERENCES rag_v7.wiki_pages(id) ON DELETE CASCADE,
    to_entity_id  BIGINT REFERENCES rag_v7.entities(id) ON DELETE SET NULL,
    anchor_text   TEXT NOT NULL,
    resolved      BOOLEAN NOT NULL DEFAULT false,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_page_id, anchor_text)
);

CREATE INDEX IF NOT EXISTS page_links_to_entity_resolved_idx
    ON rag_v7.page_links (to_entity_id, resolved);


CREATE TABLE IF NOT EXISTS rag_v7.page_sources (
    page_id       BIGINT NOT NULL REFERENCES rag_v7.wiki_pages(id) ON DELETE CASCADE,
    document_id   BIGINT NOT NULL REFERENCES rag_v7.documents(id) ON DELETE CASCADE,
    direction_key TEXT NOT NULL,
    claim_count   INT NOT NULL DEFAULT 0,
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (page_id, document_id)
);

CREATE INDEX IF NOT EXISTS page_sources_doc_idx
    ON rag_v7.page_sources (document_id);


CREATE TABLE IF NOT EXISTS rag_v7.wiki_log_entries (
    id              BIGSERIAL PRIMARY KEY,
    direction_key   TEXT NOT NULL REFERENCES rag_v7.directions(key) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_kind      rag_v7.log_event_kind NOT NULL,
    title           TEXT NOT NULL,
    ref_document_id BIGINT REFERENCES rag_v7.documents(id) ON DELETE SET NULL,
    summary         TEXT,
    affected_pages  BIGINT[] NOT NULL DEFAULT '{}',
    affected_claims BIGINT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS wiki_log_dir_ts_idx
    ON rag_v7.wiki_log_entries (direction_key, ts DESC);
