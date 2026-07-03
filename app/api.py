from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.build import build_graph


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, description="研究题目")
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


app = FastAPI(title="Tragword Research Assistant", version="0.1.0")
graph = build_graph()


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    settings = get_settings()
    state = {
        "topic": request.topic,
        "revision_count": 0,
        "approved": False,
        "review_notes": "",
        "max_revisions": (
            request.max_revisions if request.max_revisions is not None else settings.max_revisions
        ),
    }
    try:
        result = graph.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"研究流程执行失败：{exc}") from exc

    final_report = result.get("final_report") or result.get("draft", "")
    return ResearchResponse(
        topic=result["topic"],
        sub_questions=result.get("sub_questions", []),
        retrieved=result.get("retrieved", {}),
        draft=result.get("draft", ""),
        review_notes=result.get("review_notes", ""),
        approved=bool(result.get("approved", False)),
        revision_count=int(result.get("revision_count", 0)),
        max_revisions=int(result.get("max_revisions", settings.max_revisions)),
        final_report=final_report,
    )
