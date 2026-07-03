from app.config import get_settings
from app.rag.store import get_vectorstore


def retrieve_snippets(query: str, *, top_k: int | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    store = get_vectorstore()
    docs = store.similarity_search(query, k=top_k or settings.top_k)
    snippets: list[dict[str, str]] = []
    for doc in docs:
        snippets.append(
            {
                "text": doc.page_content,
                "source": str(doc.metadata.get("source", "vectorstore")),
            }
        )
    return snippets
