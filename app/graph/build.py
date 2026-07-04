from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from app.agents.planner import plan_sub_questions
from app.agents.retriever_agent import gather_for_question
from app.agents.reviewer import review_report
from app.agents.writer import write_report
from app.config import get_settings
from app.graph.state import ResearchState


def planner_node(state: ResearchState) -> dict:
    topic = state['topic']
    return {'sub_questions': plan_sub_questions(topic)}


def approve_outline_node(state: ResearchState) -> dict:
    if not state.get('require_approval'):
        return {}
    decision = interrupt(
        {
            'topic': state['topic'],
            'sub_questions': state.get('sub_questions', []),
        }
    )
    if isinstance(decision, dict):
        sub_questions = decision.get('sub_questions')
        if isinstance(sub_questions, list):
            edited = [
                item.strip()
                for item in sub_questions
                if isinstance(item, str) and item.strip()
            ]
            if edited:
                return {'sub_questions': edited}
    return {}


def fanout_retrieve(state: ResearchState) -> list[Send]:
    topic = state['topic']
    sub_questions = state.get('sub_questions') or [topic]
    return [
        Send('retrieve_one', {'topic': topic, 'question': question})
        for question in sub_questions
    ]


def retrieve_one_node(state: ResearchState) -> dict:
    topic = state['topic']
    question = state.get('question', topic)
    return {
        'question': question,
        'retrieved': {question: gather_for_question(topic, question)},
    }


def writer_node(state: ResearchState) -> dict:
    topic = state['topic']
    return {
        'draft': write_report(
            topic=topic,
            sub_questions=state.get('sub_questions', []),
            retrieved=state.get('retrieved', {}),
            review_notes=state.get('review_notes', ''),
        )
    }


def reviewer_node(state: ResearchState) -> dict:
    topic = state['topic']
    draft = state.get('draft', '')
    review = review_report(topic, draft)
    revision_count = int(state.get('revision_count', 0))
    max_revisions = int(state.get('max_revisions', get_settings().max_revisions))
    approved = bool(review.approved)
    next_revision_count = revision_count if approved else min(
        revision_count + 1,
        max_revisions,
    )
    final_report = draft if approved or next_revision_count >= max_revisions else ''
    return {
        'approved': approved,
        'review_notes': review.notes,
        'revision_count': next_revision_count,
        'final_report': final_report,
    }


def _review_route(state: ResearchState) -> str:
    if state.get('approved'):
        return 'end'
    if int(state.get('revision_count', 0)) >= int(
        state.get('max_revisions', get_settings().max_revisions)
    ):
        return 'end'
    return 'rewrite'


def build_graph(*, checkpointer=None):
    workflow: StateGraph[ResearchState] = StateGraph(ResearchState)
    workflow.add_node('planner', planner_node)
    workflow.add_node('approve_outline', approve_outline_node)
    workflow.add_node('retrieve_one', retrieve_one_node)
    workflow.add_node('writer', writer_node)
    workflow.add_node('reviewer', reviewer_node)

    workflow.add_edge(START, 'planner')
    workflow.add_edge('planner', 'approve_outline')
    workflow.add_conditional_edges('approve_outline', fanout_retrieve)
    workflow.add_edge('retrieve_one', 'writer')
    workflow.add_edge('writer', 'reviewer')
    workflow.add_conditional_edges(
        'reviewer',
        _review_route,
        {
            'rewrite': 'writer',
            'end': END,
        },
    )
    return workflow.compile(checkpointer=checkpointer)
