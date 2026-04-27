"""End-to-end проверка KnowledgeExtractor: реальный OpenAI + pgvector в Docker.

Запуск:
    OPENAI_API_KEY=sk-... .venv/bin/python try_extractor.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import psycopg
from testcontainers.postgres import PostgresContainer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_v7_wiki import WikiConfig, WikiCore  # noqa: E402
from rag_v7_wiki.providers.openai_embedder import OpenAIEmbedder  # noqa: E402
from rag_v7_wiki.providers.openai_llm import OpenAILLM  # noqa: E402

# standalone — импорт из top-level файла, без rag_v7_wiki
from knowledge_extractor import KnowledgeExtractor  # noqa: E402


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

Сам Карпати работает Director of AI at Tesla в прошлом, а сейчас руководит проектом
Eureka Labs и преподаёт курс Zero-to-Hero. Rohit Ghumare ведёт инженерное направление
в iii-engine, выступает основным автором agentmemory и отвечает за production-roll-out.
"""


def _hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


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

        # --- ingest через WikiCore (наполняем БД для последующего extract'а)
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
                row = conn.execute(
                    "SELECT status FROM rag_v7.documents WHERE id = %s", (doc_id,)
                ).fetchone()
                print(f"[doc] status={row[0]}")

        # --- extract через KnowledgeExtractor (standalone, наш модуль)
        with KnowledgeExtractor(llm, embedder, dsn) as kx:

            cases = [
                # 1. Должен матчить Андрея Карпати, инструкция про его роли
                ("Андрей Карпати", "В каких компаниях работал и кем?"),
                # 2. Должен матчить Rohit Ghumare, инструкция про его проект
                ("Rohit Ghumare", "Какой движок он построил и для чего?"),
                # 3. Negative path: бессмысленная роль, проверим graceful behaviour
                ("кошка Мурка", "Какие у этой роли задачи?"),
                # 4. Negative path: несуществующее направление
                ("Карпати", None),  # маркер для случая bad direction
            ]

            for i, (role, instr) in enumerate(cases, 1):
                _hr(f"CASE {i}: position_or_role={role!r}")
                if instr is None:
                    _hr("(этот кейс с bad direction)")
                    result = kx.extract(
                        direction_key="nope_does_not_exist",
                        position_or_role=role,
                        instruction="неважно",
                    )
                else:
                    print(f"  instruction: {instr}")
                    result = kx.extract(
                        direction_key="research",
                        position_or_role=role,
                        instruction=instr,
                    )

                print("\n--- ANSWER ---")
                print(result["answer"] or "(empty)")
                print("\n--- TRACE ---")
                # обрезаем длинные блоки чтобы лог не лопался
                trace = result["trace"]
                if len(trace) > 6000:
                    trace = trace[:6000] + "\n…(trace truncated for log)"
                print(trace)


if __name__ == "__main__":
    main()
