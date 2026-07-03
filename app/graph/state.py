from typing import TypedDict


class ResearchState(TypedDict, total=False):
    topic: str
    sub_questions: list[str]
    retrieved: dict[str, list[dict[str, str]]]
    draft: str
    review_notes: str
    approved: bool
    revision_count: int
    max_revisions: int
    final_report: str
