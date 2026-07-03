from langgraph.graph import END, StateGraph

from app.agents.planner import plan_sub_questions
from app.agents.retriever_agent import gather_evidence
from app.agents.reviewer import review_report
from app.agents.writer import write_report
from app.config import get_settings
from app.graph.state import ResearchState


def planner_node(state: ResearchState) -> dict:
    topic = state["topic"]
    return {"sub_questions": plan_sub_questions(topic)}


def retriever_node(state: ResearchState) -> dict:
    topic = state["topic"]
    sub_questions = state.get("sub_questions") or [topic]
    return {"retrieved": gather_evidence(topic, sub_questions)}


def writer_node(state: ResearchState) -> dict:
    topic = state["topic"]
    return {
        "draft": write_report(
            topic=topic,
            sub_questions=state.get("sub_questions", []),
            retrieved=state.get("retrieved", {}),
            review_notes=state.get("review_notes", ""),
        )
    }


def reviewer_node(state: ResearchState) -> dict:
    topic = state["topic"]
    draft = state.get("draft", "")
    review = review_report(topic, draft)
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", get_settings().max_revisions))
    approved = bool(review.approved)
    next_revision_count = revision_count if approved else min(revision_count + 1, max_revisions)
    final_report = draft if approved or next_revision_count >= max_revisions else ""
    return {
        "approved": approved,
        "review_notes": review.notes,
        "revision_count": next_revision_count,
        "final_report": final_report,
    }


def _review_route(state: ResearchState) -> str:
    if state.get("approved"):
        return "end"
    if int(state.get("revision_count", 0)) >= int(
        state.get("max_revisions", get_settings().max_revisions)
    ):
        return "end"
    return "rewrite"


def build_graph():
    workflow: StateGraph[ResearchState] = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        _review_route,
        {
            "rewrite": "writer",
            "end": END,
        },
    )
    return workflow.compile()
