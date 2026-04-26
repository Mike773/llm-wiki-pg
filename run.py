#!/usr/bin/env python
"""Шаблон загрузки документов и query — подмени две функции внизу секции.

Использование:

    # 1. Один раз: накатить схему
    psql -f migrations/rag_v7_schema.sql "$DATABASE_URL"

    # 2. Загрузить документ
    python run.py ingest --direction research \\
        --external-id paper-001 "Текст документа..."

    # либо из файла
    python run.py ingest --direction research \\
        --external-id paper-001 --from-file paper.txt

    # либо из stdin
    cat paper.txt | python run.py ingest --direction research

    # 3. Спросить вики
    python run.py ask --direction research "Кто такой Alice?"

    # 4. Спросить без эмбеддингов (LLM выбирает страницы по index.md)
    python run.py ask --direction research --no-embeddings "Кто такой Alice?"

DSN можно задать через --dsn или переменную окружения DATABASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypeVar

# Делаем `from rag_v7_wiki import …` рабочим без `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import psycopg
from pydantic import BaseModel

from rag_v7_wiki import WikiConfig, WikiCore, WikiQuery


# =============================================================================
# ↓↓↓  ПОДМЕНИ ЭТИ ДВЕ ФУНКЦИИ НА СВОИ  ↓↓↓
# =============================================================================

EMBEDDING_DIM = 2560  # должно совпадать с vector(N) в migrations/rag_v7_schema.sql
LLM_MODEL_NAME = "your-model-name"  # сохраняется в wiki_page_revisions.llm_model


def get_embedding(text: str) -> list[float]:
    """Верни вектор размерности EMBEDDING_DIM для одного текста.

    Например:
        client = openai.OpenAI()
        return client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=EMBEDDING_DIM,
        ).data[0].embedding
    """
    raise NotImplementedError("Подключи свой эмбеддер сюда.")


def get_llm(system: str, user: str) -> str:
    """Верни текстовый ответ LLM на пару (system, user) — обычный chat completion.

    Например:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    """
    raise NotImplementedError("Подключи свой LLM сюда.")


# =============================================================================
# ↑↑↑  ВСЁ НИЖЕ — INFRASTRUCTURE, ТРОГАТЬ НЕ НУЖНО  ↑↑↑
# =============================================================================


T = TypeVar("T", bound=BaseModel)


class _EmbedderAdapter:
    """Оборачивает get_embedding в Embedder-протокол rag_v7_wiki."""

    dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in texts]


STRUCTURED_MAX_ATTEMPTS = 3            # 1 первая + 2 retry
STRUCTURED_ECHO_RAW_CHARS = 1500       # сколько прошлого ответа LLM показать обратно
STRUCTURED_DEBUG = True                # печатать неудачные попытки в stderr


class _LLMAdapter:
    """Оборачивает get_llm в LLM-протокол.

    `complete()` — прямой проброс.
    `structured()` — JSON-mode prompting:
      - добавляем JSON-schema к user-промпту,
      - парсим ответ через _extract_json (raw_decode-based, устойчивый),
      - валидируем pydantic-ом,
      - при провале — следующая попытка получает ПОЛНЫЙ pydantic-ошибки
        и обрезанный echo собственного предыдущего ответа.
    """

    model_name = LLM_MODEL_NAME

    def complete(self, system: str, user: str) -> str:
        return get_llm(system, user)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        json_schema = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        suffix = (
            "\n\n=== STRICT OUTPUT FORMAT ===\n"
            "Верни ровно один JSON-объект, соответствующий схеме ниже. "
            "Никакого текста вне JSON. Никаких ```...``` обёрток. "
            "Все строки внутри JSON правильно экранируй (\\n, \\\", \\\\).\n"
            f"JSON Schema:\n{json_schema}"
        )

        last_error: str = ""
        last_raw: str = ""
        for attempt in range(1, STRUCTURED_MAX_ATTEMPTS + 1):
            prompt = user + suffix
            if last_error:
                prompt += (
                    "\n\n=== ПРЕДЫДУЩАЯ ПОПЫТКА БЫЛА НЕВАЛИДНОЙ ===\n"
                    "Твой прошлый ответ:\n"
                    "<<<\n"
                    f"{last_raw[:STRUCTURED_ECHO_RAW_CHARS]}"
                    f"{'…(echo обрезан)' if len(last_raw) > STRUCTURED_ECHO_RAW_CHARS else ''}\n"
                    ">>>\n\n"
                    "Ошибка валидации:\n"
                    f"{last_error}\n\n"
                    "Верни новый, ИСПРАВЛЕННЫЙ JSON, соответствующий схеме."
                )
            raw = get_llm(system, prompt)
            try:
                return schema.model_validate(_extract_json(raw))
            except Exception as exc:
                last_error = str(exc)
                last_raw = raw or ""
                if STRUCTURED_DEBUG:
                    print(
                        f"\n[_LLMAdapter.structured] attempt {attempt}/{STRUCTURED_MAX_ATTEMPTS} "
                        f"failed for {schema.__name__}:\n"
                        f"  error: {last_error[:500]}\n"
                        f"  raw  : {last_raw[:500]!r}",
                        file=sys.stderr,
                    )
        raise RuntimeError(
            f"LLM не вернул валидный {schema.__name__} за "
            f"{STRUCTURED_MAX_ATTEMPTS} попыток.\n"
            f"Последняя ошибка:\n{last_error}\n"
            f"Последний сырой ответ:\n{last_raw[:1000]!r}"
        )


def _extract_json(text: str) -> dict[str, Any]:
    """Вытаскивает первый валидный JSON-объект из ответа LLM.

    Устойчиво к: ```json fence```, тексту до/после JSON, вложенным `{`/`}`
    в строковых значениях. Использует json.JSONDecoder.raw_decode — он
    останавливается на конце первого валидного JSON, не ест лишнее.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    # 1) попытка распарсить как есть
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2) ищем первый '{', с которого raw_decode выдаёт валидный объект
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"В ответе LLM не найден JSON-объект: {text[:500]!r}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _connect_dsn(args: argparse.Namespace) -> str:
    dsn = args.dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("Не задан DSN: укажи --dsn или переменную окружения DATABASE_URL.")
    return dsn


def cmd_ingest(args: argparse.Namespace) -> None:
    dsn = _connect_dsn(args)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            content = f.read()
    elif args.text:
        content = args.text
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        sys.exit("Текст не передан: используй позиционный аргумент, --from-file или stdin.")

    if not content.strip():
        sys.exit("Пустой текст.")

    embedder = _EmbedderAdapter()
    llm = _LLMAdapter()
    config = WikiConfig(expected_embedding_dim=EMBEDDING_DIM)

    with WikiCore(db_dsn=dsn, embedder=embedder, llm=llm, config=config) as wiki:
        wiki.ensure_direction(args.direction)

        with psycopg.connect(dsn, autocommit=True) as conn:
            cur = conn.execute(
                """
                INSERT INTO rag_v7.documents
                    (direction_key, content, needs_chunking, external_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (direction_key, external_id) WHERE external_id IS NOT NULL
                DO UPDATE SET content = EXCLUDED.content,
                              needs_chunking = EXCLUDED.needs_chunking,
                              status = 'pending',
                              failed_step = NULL,
                              error = NULL
                RETURNING id;
                """,
                (
                    args.direction,
                    content,
                    args.needs_chunking or len(content) > 4000,
                    args.external_id,
                ),
            )
            doc_id = cur.fetchone()[0]

        print(f"[ingest] direction={args.direction} doc_id={doc_id} chars={len(content)}")
        wiki.process_document(args.direction, doc_id)
        print(f"[ingest] done — status='processed', wiki updated")


def cmd_pending(args: argparse.Namespace) -> None:
    dsn = _connect_dsn(args)
    embedder = _EmbedderAdapter()
    llm = _LLMAdapter()
    config = WikiConfig(expected_embedding_dim=EMBEDDING_DIM)

    with WikiCore(db_dsn=dsn, embedder=embedder, llm=llm, config=config) as wiki:
        wiki.ensure_direction(args.direction)
        processed = wiki.process_pending(args.direction, limit=args.limit)
        print(f"[pending] processed {len(processed)} docs: {processed}")


def cmd_ask(args: argparse.Namespace) -> None:
    dsn = _connect_dsn(args)
    embedder = _EmbedderAdapter()
    llm = _LLMAdapter()

    with WikiQuery(
        connection_string=dsn,
        direction_key=args.direction,
        llm=llm,
        embedder=embedder,
        use_embeddings=not args.no_embeddings,
        include_graph_expansion=args.graph,
        min_similarity=args.min_similarity,
    ) as q:
        result = q.ask(args.question)

    print("\n=== ANSWER ===")
    print(result["answer"] or "(no answer)")
    print("\n=== STAGES ===")
    for stage in result["report"]["stages"]:
        print(f"  - {stage}")
    if result["report"].get("answer_meta"):
        print(f"\n=== ANSWER META ===\n  {result['report']['answer_meta']}")
    print(f"\nelapsed_seconds={result['elapsed_seconds']:.2f}")
    if result["report"].get("errors"):
        print(f"\nERRORS: {result['report']['errors']}")


def main() -> None:
    # Общие флаги — через parents=, чтобы --dsn и --direction работали
    # как до, так и после имени subcommand-а.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (по умолчанию — env DATABASE_URL).",
    )
    common.add_argument(
        "--direction",
        default="research",
        help="ключ направления (default: research).",
    )

    parser = argparse.ArgumentParser(
        description="rag_v7_wiki ingest/query CLI шаблон",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser(
        "ingest",
        parents=[common],
        help="загрузить и обработать один документ",
    )
    p_ingest.add_argument("text", nargs="?", default=None)
    p_ingest.add_argument("--from-file", default=None)
    p_ingest.add_argument("--external-id", default=None)
    p_ingest.add_argument(
        "--needs-chunking",
        action="store_true",
        help="форсировать чанкинг (иначе авто-определение по длине).",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_pending = sub.add_parser(
        "pending",
        parents=[common],
        help="обработать все pending документы направления",
    )
    p_pending.add_argument("--limit", type=int, default=10)
    p_pending.set_defaults(func=cmd_pending)

    p_ask = sub.add_parser(
        "ask",
        parents=[common],
        help="спросить вики",
    )
    p_ask.add_argument("question")
    p_ask.add_argument("--no-embeddings", action="store_true", help="режим wiki-only (без эмбеддингов)")
    p_ask.add_argument("--graph", action="store_true", help="включить 1-hop graph expansion")
    p_ask.add_argument("--min-similarity", type=float, default=0.4)
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
