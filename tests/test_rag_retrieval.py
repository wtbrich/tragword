from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embeddings
from app.rag.retriever import retrieve_snippets
from app.rag.store import add_documents, get_vectorstore
from langchain_core.documents import Document


def test_rag_retrieval_recalls_relevant_chunk(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "milvus_local.db"
    monkeypatch.setenv("MILVUS_URI", str(db_path))
    monkeypatch.setenv("MILVUS_COLLECTION_NAME", "test_research")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # 清理缓存，确保测试使用临时目录和当前环境变量
    get_vectorstore.cache_clear()
    get_embeddings.cache_clear()

    docs = [
        Document(
            page_content="向量检索会把文本编码为向量，然后使用相似度搜索找到相关内容。",
            metadata={"source": "doc1.md"},
        ),
        Document(
            page_content="LangGraph 适合多步骤、可循环的智能体工作流编排。",
            metadata={"source": "doc2.md"},
        ),
        Document(
            page_content="传统关键词搜索依赖精确词匹配。",
            metadata={"source": "doc3.md"},
        ),
    ]

    chunks = chunk_documents(docs)
    add_documents(chunks)

    results = retrieve_snippets("什么是向量检索？", top_k=2)

    assert results
    assert any("向量检索" in item["text"] for item in results)
    assert results[0]["source"]
