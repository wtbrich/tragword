from app.config import get_settings
from app.rag.store import similarity_search


def retrieve_snippets(query: str, *, top_k: int | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    docs = similarity_search(query, top_k=top_k or settings.top_k)
    snippets: list[dict[str, str]] = []
    for doc in docs:
        snippets.append(
            {
                'text': doc.page_content,
                'source': str(doc.metadata.get('source', 'vectorstore')),
            }
        )
    return snippets
