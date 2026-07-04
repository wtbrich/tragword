from app.config import get_settings
from app.rag.hybrid import build_hybrid_retriever
from app.rag.query_expand import expand_query
from app.rag.rerank import rerank
from app.rag.store import milvus_lock, similarity_search


def retrieve_snippets(query: str, *, top_k: int | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    limit = top_k or settings.top_k
    variants = expand_query(query) if settings.multiquery_enabled else [query]
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    if settings.hybrid_enabled:
        retriever = build_hybrid_retriever(k=max(limit, settings.rerank_candidates))
        for variant in variants:
            try:
                with milvus_lock:
                    docs = retriever.invoke(variant)
            except Exception:
                docs = []
            for doc in docs:
                text = doc.page_content.strip()
                source = str(doc.metadata.get("source", "vectorstore"))
                key = (source, text)
                if not text or key in seen:
                    continue
                seen.add(key)
                merged.append({"text": text, "source": source})
    else:
        for variant in variants:
            try:
                docs = similarity_search(variant, top_k=limit)
            except Exception:
                docs = []
            for doc in docs:
                text = doc.page_content.strip()
                source = str(doc.metadata.get("source", "vectorstore"))
                key = (source, text)
                if not text or key in seen:
                    continue
                seen.add(key)
                merged.append({"text": text, "source": source})

    if not merged:
        return []

    if settings.rerank_enabled:
        return rerank(query, merged[: max(limit, settings.rerank_candidates)], limit)

    for index, item in enumerate(merged[:limit], start=1):
        item.setdefault("score", f"{1.0 / index:.6f}")
    return merged[:limit]
