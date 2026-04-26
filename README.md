# rag_v7_wiki — LLM Wiki on Postgres + pgvector

An ingest-time LLM-maintained knowledge base. Documents go in; a structured,
interlinked wiki comes out — automatically updated, deduplicated, and
versioned in Postgres. Inspired by Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
and the [v2 extensions](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2)
by Rohit Ghumare.

The compounding work — summarising, cross-referencing, contradiction
detection, predicate normalisation — happens **once, at ingest**, not on
every query. The retrieval layer (BM25 / hybrid search / lint passes) is
deliberately out of scope for this repo.

## Pipeline

Each document goes through 11 steps inside `WikiCore.process_document`:

1. **redact** — regex-based PII / API-key scrubbing (`[REDACTED:kind:n]`),
   audit trail in `documents.redactions`.
2. **chunk** + optional **summary** for long documents.
3. **embed** chunks (and summaries) into pgvector.
4. **extract entities** with type, aliases, salient attrs, supporting chunks.
5. **resolve entities** via kNN cosine + LLM arbiter, deduped by canonical name.
6. **extract claims** as atomic `(subject, predicate, object)` triples
   with citations.
7. **normalise predicates** into a controlled per-direction vocabulary
   (`canonical_predicates` table).
8. **supersession + contradiction auto-resolution** — LLM arbiter decides
   `same / supersedes_old / contradiction / orthogonal`. Confident
   contradictions are auto-resolved (`decided_by='auto_arbiter'`),
   uncertain ones stay flagged.
9. **tier promotion** — claims walk `working → episodic → semantic` based
   on confirmation count and age.
10. **synthesise pages** — one per source document (`page_kind='source'`)
    plus one per affected entity, each through a quality-score pass with
    optional re-synthesis. Coverage metrics + provenance written to
    `wiki_pages` and `page_sources`.
11. **rebuild singleton pages** — `index` (catalogue grouped by page_kind)
    and `log` (chronological `## [YYYY-MM-DD] ingest | title` entries),
    both materialised as wiki pages from underlying tables.

Confidence updates use a Bayesian noisy-OR rollup
(`new = 1 − (1−old)·(1−hint)`) so each confirmation moves a claim
asymptotically toward 1.

## Layout

```
migrations/rag_v7_schema.sql      — single-file idempotent schema
src/rag_v7_wiki/
    core.py                       — WikiCore: pipeline orchestration + public read API
    config.py                     — WikiConfig: thresholds, pool sizes, tier rules
    schemas.py                    — pydantic LLM I/O schemas
    protocols.py                  — Embedder / LLM protocols
    dao/                          — psycopg + pgvector data access
        documents.py  chunks.py  entities.py  claims.py
        pages.py      log.py     predicates.py
    pipeline/                     — one module per ingest step
        redact.py        chunking.py      embedding.py
        entities.py      claims.py        predicates.py
        contradictions.py  sources.py    synthesis.py
        quality.py       coverage.py      linking.py
        indexing.py      log.py
    providers/                    — concrete LLM / embedder backends
tests/                            — testcontainers-driven Postgres tests
e2e_check.py                      — manual end-to-end with real OpenAI keys
```

## Quick start

Requires Docker (testcontainers spins up `pgvector/pgvector:pg16`).

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
pytest                               # 20 tests, ~5s
```

For a real end-to-end run:

```bash
OPENAI_API_KEY=sk-... .venv/bin/python e2e_check.py
```

## Public API

```python
from rag_v7_wiki import WikiCore, WikiConfig

with WikiCore(db_dsn=DSN, embedder=..., llm=..., config=WikiConfig()) as wiki:
    wiki.ensure_direction("research")
    wiki.process_document("research", document_id)

    print(wiki.get_index_md("research"))         # rendered index page
    print(wiki.get_log_md("research"))           # chronological log
    src_page = wiki.get_source_page("research", document_id)
    entries  = wiki.list_log_entries("research", limit=20)
```

## Status

WIP. The ingest side is feature-complete; query/retrieval (BM25, hybrid
search, lint, scheduled decay) is intentionally not yet implemented — see
the [LLM Wiki v2 spec](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2)
for what comes next.

## License

MIT — see [LICENSE](LICENSE).
