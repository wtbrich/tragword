from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.config import get_settings
from app.graph.build import build_graph
from app.graph.state import ResearchState, merge_retrieved


def _merge_state(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    if 'retrieved' in patch:
        merged['retrieved'] = merge_retrieved(
            merged.get('retrieved', {}),
            patch.get('retrieved', {}),
        )
    for key, value in patch.items():
        if key == 'retrieved':
            continue
        merged[key] = value
    return merged


def _final_event(state: dict[str, Any]) -> dict[str, Any]:
    final_report = str(state.get('final_report') or state.get('draft', ''))
    return {
        'node': 'final',
        'detail': {
            'approved': bool(state.get('approved', False)),
            'revision_count': int(state.get('revision_count', 0)),
            'final_report': final_report,
        },
    }


def format_update_event(node: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not payload or node in {'approve_outline'} or node.startswith('__'):
        return None
    if node == 'planner':
        return {
            'node': node,
            'detail': {
                'count': len(payload.get('sub_questions', [])),
                'sub_questions': list(payload.get('sub_questions', [])),
            },
        }
    if node == 'retrieve_one':
        retrieved = payload.get('retrieved', {})
        question = payload.get('question', '')
        return {
            'node': node,
            'detail': {
                'question': question,
                'evidence_count': len(retrieved.get(question, [])) if question else 0,
            },
        }
    if node == 'writer':
        draft = str(payload.get('draft', ''))
        return {'node': node, 'detail': {'draft_length': len(draft)}}
    if node == 'reviewer':
        return {
            'node': node,
            'detail': {
                'approved': bool(payload.get('approved', False)),
                'revision_count': int(payload.get('revision_count', 0)),
            },
        }
    return {'node': node, 'detail': {'keys': sorted(payload.keys())}}


def stream_graph_events(
    graph,
    inputs,
    *,
    initial_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    current: dict[str, Any] = dict(initial_state)
    for update in graph.stream(inputs, config, stream_mode='updates'):
        if not isinstance(update, dict):
            continue
        for node, payload in update.items():
            if not isinstance(payload, dict):
                continue
            current = _merge_state(current, payload)
            event = format_update_event(node, payload)
            if event is not None:
                yield event
    yield _final_event(current)


def stream_research(topic: str, *, max_revisions: int | None = None) -> Iterator[dict[str, Any]]:
    settings = get_settings()
    state: ResearchState = {
        'topic': topic,
        'revision_count': 0,
        'approved': False,
        'review_notes': '',
        'max_revisions': max_revisions if max_revisions is not None else settings.max_revisions,
    }
    graph = build_graph()
    yield from stream_graph_events(graph, state, initial_state=state)
