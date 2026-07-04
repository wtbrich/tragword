from app.rag import query_expand as query_expand_module
from app.rag.query_expand import expand_query


def test_query_expand_falls_back_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv('MULTIQUERY_ENABLED', 'true')

    def raise_error():
        raise Exception('no llm available')

    monkeypatch.setattr(query_expand_module, 'get_chat_model', raise_error)

    query = '向量检索是什么'
    assert expand_query(query) == [query]
