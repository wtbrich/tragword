from app.graph.build import build_graph
from app.graph.state import merge_retrieved


class DummyReview:
    def __init__(self, approved: bool = True, notes: str = 'ok') -> None:
        self.approved = approved
        self.notes = notes


def test_merge_retrieved_updates_by_question() -> None:
    left = {'q1': [{'text': 'a', 'source': 's1'}]}
    right = {
        'q2': [{'text': 'b', 'source': 's2'}],
        'q1': [{'text': 'c', 'source': 's3'}],
    }

    merged = merge_retrieved(left, right)

    assert merged == {
        'q1': [{'text': 'c', 'source': 's3'}],
        'q2': [{'text': 'b', 'source': 's2'}],
    }


def test_graph_fanout_merges_parallel_retrieval(monkeypatch) -> None:
    from app.graph import build as build_module

    monkeypatch.setattr(
        build_module,
        'plan_sub_questions',
        lambda topic: [
            f'{topic} 背景是什么？',
            f'{topic} 关键步骤是什么？',
        ],
    )
    monkeypatch.setattr(
        build_module,
        'gather_for_question',
        lambda topic, question: [
            {'text': f'{topic}::{question}', 'source': question},
        ],
    )
    monkeypatch.setattr(build_module, 'write_report', lambda **kwargs: 'draft [1]')
    monkeypatch.setattr(
        build_module,
        'review_report',
        lambda topic, draft: DummyReview(approved=True, notes='approved'),
    )

    graph = build_graph()
    result = graph.invoke(
        {
            'topic': 'LangGraph',
            'revision_count': 0,
            'approved': False,
            'review_notes': '',
            'max_revisions': 1,
        }
    )

    assert set(result['retrieved']) == {
        'LangGraph 背景是什么？',
        'LangGraph 关键步骤是什么？',
    }
    assert all(
        result['retrieved'][question][0]['source'] == question
        for question in result['retrieved']
    )
    assert result['final_report'] == 'draft [1]'
