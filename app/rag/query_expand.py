from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm import get_chat_model


class QueryExpansionResult(BaseModel):
    queries: list[str] = Field(min_length=1)


def expand_query(query: str) -> list[str]:
    query = query.strip()
    if not query:
        return []
    try:
        llm = get_chat_model().with_structured_output(
            QueryExpansionResult,
            method='function_calling',
        )
        result = llm.invoke(
            [
                (
                    'system',
                    '你是检索查询改写器。请基于原始问题生成 2 到 3 个等价或近义检索表达，'
                    '便于提升召回。返回的列表必须包含原始问题。',
                ),
                ('human', f'原始问题：{query}'),
            ]
        )
        queries: list[str] = []
        for item in result.queries:
            text = item.strip()
            if text:
                queries.append(text)
        if query not in queries:
            queries.insert(0, query)
        return queries or [query]
    except Exception:
        return [query]
