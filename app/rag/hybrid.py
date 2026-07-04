from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app.config import get_settings
from app.rag.store import get_vectorstore, load_persisted_chunks


def build_hybrid_retriever(*, k: int | None = None) -> object:
    settings = get_settings()
    top_k = k or settings.top_k
    vector_retriever = get_vectorstore().as_retriever(search_kwargs={'k': top_k})
    chunks = load_persisted_chunks()
    if not chunks:
        return vector_retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])
