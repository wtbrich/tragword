from app.graph.stream import stream_research


class DummyReview:
    def __init__(self, approved: bool = True, notes: str = 'ok') -> None:
        self.approved = approved
        self.notes = notes


def test_stream_research_emits_progress_events(monkeypatch) -> None:
    from app.graph import build as build_module

    monkeypatch.setattr(
        build_module,
        'plan_sub_questions',
        lambda topic: [f'{topic} 背景', f'{topic} 实践'],
    )
    monkeypatch.setattr(
        build_module,
        'gather_for_question',
        lambda topic, question: [
            {'text': f'{question} evidence', 'source': question},
        ],
    )
    monkeypatch.setattr(build_module, 'write_report', lambda **kwargs: 'draft [1]')
    monkeypatch.setattr(
        build_module,
        'review_report',
        lambda topic, draft: DummyReview(approved=True, notes='approved'),
    )

    events = list(stream_research('LangGraph'))
    nodes = [event['node'] for event in events]

    assert nodes[0] == 'planner'
    assert nodes.count('retrieve_one') == 2
    assert 'writer' in nodes
    assert 'reviewer' in nodes
    assert nodes[-1] == 'final'
    assert events[-1]['detail']['final_report'] == 'draft [1]'
