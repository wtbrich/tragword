from typing import Annotated, TypedDict


def merge_retrieved(
    left: dict[str, list[dict[str, str]]],
    right: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    merged = dict(left)
    merged.update(right)
    return merged


def merge_question(left: str | None, right: str | None) -> str | None:
    return right if right is not None else left


class ResearchState(TypedDict, total=False):
    topic: str
    require_approval: bool
    sub_questions: list[str]
    question: Annotated[str | None, merge_question]
    retrieved: Annotated[
        dict[str, list[dict[str, str]]],
        merge_retrieved,
    ]
    draft: str
    review_notes: str
    approved: bool
    revision_count: int
    max_revisions: int
    final_report: str
