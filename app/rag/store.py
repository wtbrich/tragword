from contextlib import suppress
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, connections

from app.config import get_settings
from app.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Milvus:
    settings = get_settings()
    uri = Path(settings.milvus_uri)
    uri.parent.mkdir(parents=True, exist_ok=True)
    probe_client = MilvusClient(uri=str(uri))
    with suppress(Exception):
        connections.connect(alias=probe_client._using, uri=str(uri))
    store = Milvus(
        embedding_function=get_embeddings(),
        collection_name=settings.milvus_collection_name,
        connection_args={"uri": str(uri)},
        auto_id=True,
    )
    with suppress(Exception):
        connections.connect(alias=store.alias, uri=str(uri))
    return store


def add_documents(documents: list[Document]) -> int:
    if not documents:
        return 0
    store = get_vectorstore()
    store.add_documents(documents)
    return len(documents)


def load_documents_from_directory(directory: str | Path) -> list[Document]:
    directory_path = Path(directory)
    documents: list[Document] = []
    for path in sorted(directory_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)},
            )
        )
    return documents
