from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from threading import RLock
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.config import get_settings
from app.graph.build import build_graph
from app.graph.stream import stream_graph_events

_settings = get_settings()
_checkpoint_path = Path(_settings.checkpoint_db)
_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
_checkpoint_conn = sqlite3.connect(_checkpoint_path, check_same_thread=False)
_checkpoint_lock = RLock()
interactive_graph = build_graph(checkpointer=SqliteSaver(_checkpoint_conn))


def _config(thread_id: str) -> dict[str, Any]:
    return {'configurable': {'thread_id': thread_id}}


def start_research(
    topic: str,
    *,
    thread_id: str | None = None,
    max_revisions: int | None = None,
) -> dict[str, Any]:
    thread_id = thread_id or str(uuid.uuid4())
    config = _config(thread_id)
    state = {
        'topic': topic,
        'revision_count': 0,
        'approved': False,
        'review_notes': '',
        'max_revisions': max_revisions if max_revisions is not None else _settings.max_revisions,
        'require_approval': True,
    }
    with _checkpoint_lock:
        interactive_graph.invoke(state, config)
        snapshot = interactive_graph.get_state(config)
    return {
        'thread_id': thread_id,
        'status': 'awaiting_approval' if snapshot.next else 'running',
        'topic': snapshot.values.get('topic', topic),
        'sub_questions': list(snapshot.values.get('sub_questions', [])),
    }


def resume_research(
    thread_id: str,
    *,
    approved: bool,
    sub_questions: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    config = _config(thread_id)
    with _checkpoint_lock:
        snapshot = interactive_graph.get_state(config)
        initial_state = dict(snapshot.values)
        command = Command(resume={'approved': approved, 'sub_questions': sub_questions})
        yield from stream_graph_events(
            interactive_graph,
            command,
            initial_state=initial_state,
            config=config,
        )
