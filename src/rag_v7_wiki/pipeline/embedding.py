from __future__ import annotations

from rag_v7_wiki.dao.chunks import ChunkDAO
from rag_v7_wiki.dao.documents import DocumentDAO
from rag_v7_wiki.protocols import Embedder


def embed_document_chunks(
    direction_key: str,
    document_id: int,
    embedder: Embedder,
    chunk_dao: ChunkDAO,
    batch_size: int = 64,
) -> None:
    pending = chunk_dao.needing_embedding(direction_key, document_id)
    if not pending:
        return
    ids = [c["id"] for c in pending]
    texts = [c["content"] for c in pending]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        vectors = embedder.embed(batch_texts)
        chunk_dao.set_embeddings(direction_key, batch_ids, vectors)


def embed_summary(
    direction_key: str,
    document_id: int,
    summary: str,
    embedder: Embedder,
    document_dao: DocumentDAO,
) -> None:
    if not summary:
        return
    [vec] = embedder.embed([summary])
    document_dao.set_summary(direction_key, document_id, summary, vec)
