from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.build import build_graph
from app.graph.interactive import resume_research, start_research
from app.graph.stream import stream_research


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, description='研究题目')
    max_revisions: int | None = Field(default=None, ge=0)


class ResearchResponse(BaseModel):
    topic: str
    sub_questions: list[str]
    retrieved: dict[str, list[dict[str, str]]]
    draft: str
    review_notes: str
    approved: bool
    revision_count: int
    max_revisions: int
    final_report: str


class InteractiveStartRequest(BaseModel):
    topic: str = Field(min_length=1)
    max_revisions: int | None = Field(default=None, ge=0)
    thread_id: str | None = None


class InteractiveResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    sub_questions: list[str] | None = None


app = FastAPI(title='Tragword Research Assistant', version='0.1.0')
graph = build_graph()


def _initial_state(request: ResearchRequest) -> dict[str, object]:
    settings = get_settings()
    return {
        'topic': request.topic,
        'revision_count': 0,
        'approved': False,
        'review_notes': '',
        'max_revisions': (
            request.max_revisions
            if request.max_revisions is not None
            else settings.max_revisions
        ),
    }


def _sse_events(events: Iterator[dict[str, object]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post('/research', response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    settings = get_settings()
    try:
        result = graph.invoke(_initial_state(request))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'研究流程执行失败：{exc}') from exc

    final_report = result.get('final_report') or result.get('draft', '')
    return ResearchResponse(
        topic=result['topic'],
        sub_questions=result.get('sub_questions', []),
        retrieved=result.get('retrieved', {}),
        draft=result.get('draft', ''),
        review_notes=result.get('review_notes', ''),
        approved=bool(result.get('approved', False)),
        revision_count=int(result.get('revision_count', 0)),
        max_revisions=int(result.get('max_revisions', settings.max_revisions)),
        final_report=final_report,
    )


@app.post('/research/stream')
def research_stream(request: ResearchRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        yield from _sse_events(
            stream_research(request.topic, max_revisions=request.max_revisions)
        )

    return StreamingResponse(event_stream(), media_type='text/event-stream')


@app.post('/research/interactive/start')
def research_interactive_start(request: InteractiveStartRequest) -> dict[str, object]:
    return start_research(
        request.topic,
        thread_id=request.thread_id,
        max_revisions=request.max_revisions,
    )


@app.post('/research/interactive/resume')
def research_interactive_resume(request: InteractiveResumeRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        yield from _sse_events(
            resume_research(
                request.thread_id,
                approved=request.approved,
                sub_questions=request.sub_questions,
            )
        )

    return StreamingResponse(event_stream(), media_type='text/event-stream')
