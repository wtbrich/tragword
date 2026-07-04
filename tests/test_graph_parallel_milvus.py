from app.config import get_settings
from app.graph.build import build_graph
from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embeddings
from app.rag.store import add_documents, get_vectorstore
from langchain_core.documents import Document


def test_graph_parallel_retrieval_with_real_milvus(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / 'milvus_local.db'
    monkeypatch.setenv('MILVUS_DB_URI', str(db_path))
    monkeypatch.setenv('MILVUS_COLLECTION_NAME', 'parallel_research')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'huggingface')
    monkeypatch.setenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
    monkeypatch.setenv('SEARCH_ENABLED', 'false')
    get_settings.cache_clear()
    get_vectorstore.cache_clear()
    get_embeddings.cache_clear()

    docs = [
        Document(
            page_content='LangGraph 使用 Send 可以并行分发子任务。',
            metadata={'source': 'send.md'},
        ),
        Document(
            page_content='Milvus Lite 本地模式适合离线测试，但并发访问需要串行化。',
            metadata={'source': 'milvus.md'},
        ),
        Document(
            page_content='研究助手需要在写作时整合多个子问题的证据。',
            metadata={'source': 'writer.md'},
        ),
    ]
    add_documents(chunk_documents(docs))
    get_vectorstore.cache_clear()

    graph = build_graph()
    result = graph.invoke(
        {
            'topic': 'LangGraph 并行检索',
            'revision_count': 0,
            'approved': False,
            'review_notes': '',
            'max_revisions': 1,
        }
    )

    planned = result['sub_questions']
    assert planned
    assert set(result['retrieved']) == set(planned)
    assert all(result['retrieved'][question] for question in planned)
    assert result['final_report']
