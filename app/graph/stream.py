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


def _event(node: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {'node': node, 'detail': detail}


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
    current: dict[str, Any] = dict(state)

    for update in graph.stream(state, stream_mode='updates'):
        if not isinstance(update, dict):
            continue
        for node, payload in update.items():
            if not isinstance(payload, dict):
                continue
            if node == 'planner':
                detail = {
                    'count': len(payload.get('sub_questions', [])),
                    'sub_questions': list(payload.get('sub_questions', [])),
                }
            elif node == 'retrieve_one':
                retrieved = payload.get('retrieved', {})
                question = payload.get('question', '')
                detail = {
                    'question': question,
                    'evidence_count': len(retrieved.get(question, [])) if question else 0,
                }
            elif node == 'writer':
                draft = str(payload.get('draft', ''))
                detail = {'draft_length': len(draft)}
            elif node == 'reviewer':
                detail = {
                    'approved': bool(payload.get('approved', False)),
                    'revision_count': int(payload.get('revision_count', 0)),
                }
            else:
                detail = {'keys': sorted(payload.keys())}
            current = _merge_state(current, payload)
            yield _event(node, detail)

    final_report = str(current.get('final_report') or current.get('draft', ''))
    yield _event(
        'final',
        {
            'approved': bool(current.get('approved', False)),
            'revision_count': int(current.get('revision_count', 0)),
            'final_report': final_report,
        },
    )
