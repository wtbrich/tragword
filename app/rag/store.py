import json
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from threading import RLock

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, connections

from app.config import get_settings
from app.rag.embeddings import get_embeddings

# Milvus Lite local mode is not safe under concurrent access; serialize every
# Milvus touch so the parallel retrieval fan-out cannot open/query it at once.
milvus_lock = RLock()


def _chunks_sidecar_path() -> Path | None:
    uri = get_settings().milvus_uri.strip()
    if uri.endswith(".db"):
        return Path(uri).with_name("chunks.jsonl")
    return None


def persist_chunks(documents: list[Document]) -> None:
    sidecar = _chunks_sidecar_path()
    if sidecar is None or not documents:
        return
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("a", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            handle.write("\n")


def load_persisted_chunks() -> list[Document]:
    sidecar = _chunks_sidecar_path()
    if sidecar is None or not sidecar.exists():
        return []
    documents: list[Document] = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        page_content = str(payload.get("page_content", "")).strip()
        if not page_content:
            continue
        metadata = payload.get("metadata")
        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return documents


def _disconnect_milvus_connections() -> None:
    with suppress(Exception):
        for alias, conn in connections.list_connections():
            if conn is None:
                continue
            with suppress(Exception):
                connections.disconnect(alias)


def reset_vectorstore() -> None:
    with milvus_lock:
        get_vectorstore.cache_clear()
        _disconnect_milvus_connections()


@lru_cache(maxsize=1)
def get_vectorstore() -> Milvus:
    settings = get_settings()
    uri = Path(settings.milvus_uri)
    uri.parent.mkdir(parents=True, exist_ok=True)
    with milvus_lock:
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
    with milvus_lock:
        store = get_vectorstore()
        store.add_documents(documents)
    persist_chunks(documents)
    return len(documents)


def similarity_search(query: str, *, top_k: int) -> list[Document]:
    with milvus_lock:
        return get_vectorstore().similarity_search(query, k=top_k)


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
