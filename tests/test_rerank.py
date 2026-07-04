from app.rag import rerank as rerank_module
from app.rag.rerank import rerank


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for _, text in pairs:
            if '最相关答案' in text:
                scores.append(0.99)
            elif '次相关' in text:
                scores.append(0.5)
            else:
                scores.append(0.1)
        return scores


def test_rerank_puts_relevant_snippet_first(monkeypatch) -> None:
    monkeypatch.setattr(rerank_module, 'get_cross_encoder', lambda: FakeCrossEncoder())
    snippets = [
        {'text': '这是一段次相关说明', 'source': 'a'},
        {'text': '这里包含最相关答案以及关键词', 'source': 'b'},
        {'text': '无关内容', 'source': 'c'},
    ]

    ranked = rerank('最相关答案是什么', snippets, top_k=3)

    assert ranked
    assert ranked[0]['source'] == 'b'
    assert ranked[0]['score'] != '0.000000'
