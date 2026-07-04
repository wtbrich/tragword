from app.config import get_settings
from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embeddings
from app.rag.retriever import retrieve_snippets
from app.rag.store import add_documents, get_vectorstore
from langchain_core.documents import Document


def test_hybrid_retrieval_uses_bm25_exact_keyword(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'milvus_local.db'
    monkeypatch.setenv('MILVUS_DB_URI', str(db_path))
    monkeypatch.setenv('MILVUS_COLLECTION_NAME', 'hybrid_research')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'huggingface')
    monkeypatch.setenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
    monkeypatch.setenv('HYBRID_ENABLED', 'true')
    monkeypatch.setenv('RERANK_ENABLED', 'false')
    monkeypatch.setenv('MULTIQUERY_ENABLED', 'false')
    get_settings.cache_clear()
    get_vectorstore.cache_clear()
    get_embeddings.cache_clear()

    docs = [
        Document(
            page_content='向量检索适合语义相似的内容召回。',
            metadata={'source': 'semantic.md'},
        ),
        Document(
            page_content='BM25_ONLY_TOKEN_2026 是一个只靠关键词就能命中的稀有标记。',
            metadata={'source': 'keyword.md'},
        ),
        Document(
            page_content='回炉重写有助于提升报告质量。',
            metadata={'source': 'report.md'},
        ),
    ]

    chunks = chunk_documents(docs)
    add_documents(chunks)

    results = retrieve_snippets('BM25_ONLY_TOKEN_2026 是什么？', top_k=3)

    assert results
    assert any(item['source'] == 'keyword.md' for item in results)
    assert any('BM25_ONLY_TOKEN_2026' in item['text'] for item in results)
