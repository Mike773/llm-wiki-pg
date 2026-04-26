"""Ad hoc e2e: реальные OpenAI вызовы + Postgres+pgvector через testcontainers.

Запуск (api-key через env):
    OPENAI_API_KEY=sk-... .venv/bin/python e2e_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from testcontainers.postgres import PostgresContainer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_v7_wiki import WikiConfig, WikiCore  # noqa: E402
from rag_v7_wiki.providers.openai_embedder import OpenAIEmbedder  # noqa: E402
from rag_v7_wiki.providers.openai_llm import OpenAILLM  # noqa: E402


SAMPLE_DOC = """\
LLM Wiki — это паттерн персонального управления знаниями, описанный Андреем Карпати
в апреле 2026 года. Идея в том, что LLM-агент активно курирует структурированную
markdown-вики, а не просто извлекает фрагменты из сырых документов на лету.

Архитектура состоит из трёх слоёв: raw sources (неизменяемые оригиналы),
wiki (генерируемые LLM markdown-страницы) и schema (конфиг, описывающий структуру).

Поверх паттерна Rohit Ghumare построил agentmemory — production-движок, который
формализует уроки v2: confidence scoring, supersession, иерархическая память
из четырёх уровней (working, episodic, semantic, procedural), типизированный
knowledge graph и hybrid search через RRF.

Ключевое отличие от классического RAG: тяжёлая работа синтеза переносится на
ingest-time — один раз при загрузке документа, а не на каждый запрос.
"""


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY не задан", file=sys.stderr)
        sys.exit(2)

    migration = (Path(__file__).parent / "migrations" / "rag_v7_schema.sql").read_text()

    container = PostgresContainer(
        "pgvector/pgvector:pg16",
        username="rag",
        password="rag",
        dbname="rag",
    )
    with container as pg:
        host = pg.get_container_host_ip()
        port = pg.get_exposed_port(5432)
        dsn = f"postgresql://rag:rag@{host}:{port}/rag"
        print(f"[pg] {dsn}")

        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(migration)
        print("[pg] migration applied")

        embedder = OpenAIEmbedder(model="text-embedding-3-large", dim=2560)
        llm = OpenAILLM(model="gpt-4o-mini")

        with WikiCore(
            db_dsn=dsn, embedder=embedder, llm=llm, config=WikiConfig()
        ) as wiki:
            wiki.ensure_direction(
                "research",
                name="Research notes",
                description="LLM Wiki / agentmemory",
            )
            with psycopg.connect(dsn, autocommit=True) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO rag_v7.documents (direction_key, content, needs_chunking)
                    VALUES ('research', %s, false)
                    RETURNING id;
                    """,
                    (SAMPLE_DOC,),
                )
                doc_id = cur.fetchone()[0]
            print(f"[doc] inserted id={doc_id}")

            print("[pipeline] processing...")
            wiki.process_document("research", doc_id)
            print("[pipeline] done")

            with psycopg.connect(dsn, autocommit=True) as conn:
                print("\n=== Document status ===")
                row = conn.execute(
                    "SELECT status, failed_step, error FROM rag_v7.documents WHERE id = %s",
                    (doc_id,),
                ).fetchone()
                print(f"status={row[0]} failed_step={row[1]} error={row[2]}")

                print("\n=== Entities ===")
                for r in conn.execute(
                    "SELECT entity_type, canonical_name, mention_count, salient_attrs "
                    "FROM rag_v7.entities WHERE direction_key = 'research' ORDER BY id"
                ):
                    print(f"  {r[0]:20s} | {r[1]:40s} | mentions={r[2]} | {r[3]}")

                print("\n=== Claims (active) ===")
                for r in conn.execute(
                    """
                    SELECT e.canonical_name, c.predicate,
                           COALESCE(o.canonical_name, c.object_text),
                           c.claim_text, c.confidence, c.times_confirmed
                    FROM rag_v7.claims c
                    JOIN rag_v7.entities e ON e.id = c.subject_entity_id
                    LEFT JOIN rag_v7.entities o ON o.id = c.object_entity_id
                    WHERE c.direction_key = 'research' AND c.status = 'active'
                    ORDER BY c.id;
                    """
                ):
                    print(f"  [{r[0]}] —{r[1]}→ {r[2]}")
                    print(f"      «{r[3]}» (conf={r[4]:.2f}, ×{r[5]})")

                print("\n=== Wiki pages ===")
                for r in conn.execute(
                    """
                    SELECT title, slug, version, length(content_md)
                    FROM rag_v7.wiki_pages
                    WHERE direction_key = 'research'
                    ORDER BY id;
                    """
                ):
                    print(f"  {r[0]} (slug={r[1]}, v{r[2]}, {r[3]} chars)")

                print("\n=== Sample page content ===")
                page = conn.execute(
                    "SELECT title, content_md FROM rag_v7.wiki_pages "
                    "WHERE direction_key = 'research' ORDER BY id LIMIT 1"
                ).fetchone()
                if page:
                    print(f"--- {page[0]} ---")
                    print(page[1])

                print("\n=== Page links ===")
                for r in conn.execute(
                    """
                    SELECT p.title, pl.anchor_text, pl.resolved,
                           e.canonical_name
                    FROM rag_v7.page_links pl
                    JOIN rag_v7.wiki_pages p ON p.id = pl.from_page_id
                    LEFT JOIN rag_v7.entities e ON e.id = pl.to_entity_id
                    WHERE pl.direction_key = 'research'
                    ORDER BY pl.id;
                    """
                ):
                    arrow = "→" if r[2] else "✗"
                    print(f"  {r[0]:30s} [[{r[1]}]] {arrow} {r[3] or 'BROKEN'}")


if __name__ == "__main__":
    main()
