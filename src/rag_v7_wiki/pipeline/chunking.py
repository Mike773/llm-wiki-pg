from __future__ import annotations

from semantic_text_splitter import TextSplitter

from rag_v7_wiki.protocols import LLM
from rag_v7_wiki.schemas import DocumentSummaryResponse


def chunk_document(
    content: str,
    needs_chunking: bool,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[tuple[int, str, int]]:
    """Делит документ на чанки.

    Если needs_chunking=False — один «логический чанк» = весь content.
    Иначе семантический сплит по символам через semantic-text-splitter.
    Возвращает list[(ord, content, length_in_chars)].
    """
    if not needs_chunking:
        return [(0, content, len(content))]

    splitter = TextSplitter(
        capacity=chunk_size_chars,
        overlap=chunk_overlap_chars,
    )
    pieces = splitter.chunks(content)
    return [(i, piece, len(piece)) for i, piece in enumerate(pieces)]


SUMMARY_SYSTEM = (
    "Ты составляешь краткую фактическую сводку документа на 200–500 слов. "
    "Сосредоточься на сущностях, событиях и утверждениях, которые понадобятся "
    "для последующего извлечения. Не выдумывай ничего, чего нет в тексте."
)


def summarize_document(content: str, llm: LLM) -> str:
    response = llm.structured(
        system=SUMMARY_SYSTEM,
        user=f"Документ:\n\n{content}",
        schema=DocumentSummaryResponse,
    )
    return response.summary.strip()
