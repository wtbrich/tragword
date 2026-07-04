from app.agents.reviewer import ReviewResult
from app.graph.build import build_graph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class DummyReview:
    def __init__(self, approved: bool = True, notes: str = 'ok') -> None:
        self.approved = approved
        self.notes = notes


def test_outline_interrupt_and_resume_with_edits(monkeypatch) -> None:
    from app.graph import build as build_module

    monkeypatch.setattr(
        build_module,
        'plan_sub_questions',
        lambda topic: [f'{topic} 背景', f'{topic} 实践'],
    )
    monkeypatch.setattr(
        build_module,
        'gather_for_question',
        lambda topic, question: [{'text': f'{topic}::{question}', 'source': question}],
    )
    monkeypatch.setattr(
        build_module,
        'write_report',
        lambda **kwargs: 'draft:' + '|'.join(kwargs['sub_questions']),
    )
    monkeypatch.setattr(
        build_module,
        'review_report',
        lambda topic, draft: ReviewResult(approved=True, notes='approved'),
    )

    graph = build_graph(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'thread-edit'}}

    first = graph.invoke(
        {
            'topic': 'LangGraph HITL',
            'revision_count': 0,
            'approved': False,
            'review_notes': '',
            'max_revisions': 1,
            'require_approval': True,
        },
        config,
    )
    assert '__interrupt__' in first

    snapshot = graph.get_state(config)
    assert snapshot.next == ('approve_outline',)
    assert snapshot.values['sub_questions'] == ['LangGraph HITL 背景', 'LangGraph HITL 实践']

    edited = ['LangGraph HITL 背景（编辑）', 'LangGraph HITL 实践（编辑）']
    result = graph.invoke(
        Command(resume={'approved': True, 'sub_questions': edited}),
        config,
    )

    assert result['sub_questions'] == edited
    assert '（编辑）' in result['final_report']


def test_outline_interrupt_approval_keeps_original_outline(monkeypatch) -> None:
    from app.graph import build as build_module

    monkeypatch.setattr(
        build_module,
        'plan_sub_questions',
        lambda topic: [f'{topic} 背景', f'{topic} 实践'],
    )
    monkeypatch.setattr(
        build_module,
        'gather_for_question',
        lambda topic, question: [{'text': f'{topic}::{question}', 'source': question}],
    )
    monkeypatch.setattr(
        build_module,
        'write_report',
        lambda **kwargs: 'draft:' + '|'.join(kwargs['sub_questions']),
    )
    monkeypatch.setattr(
        build_module,
        'review_report',
        lambda topic, draft: DummyReview(approved=True, notes='approved'),
    )

    graph = build_graph(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'thread-approve'}}

    graph.invoke(
        {
            'topic': 'LangGraph HITL',
            'revision_count': 0,
            'approved': False,
            'review_notes': '',
            'max_revisions': 1,
            'require_approval': True,
        },
        config,
    )

    result = graph.invoke(Command(resume={'approved': True}), config)

    assert result['sub_questions'] == ['LangGraph HITL 背景', 'LangGraph HITL 实践']
    assert result['final_report'] == 'draft:LangGraph HITL 背景|LangGraph HITL 实践'
