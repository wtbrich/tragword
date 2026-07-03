from pydantic import BaseModel, Field

from app.llm import get_chat_model


class ReviewResult(BaseModel):
    approved: bool = Field(description="是否通过")
    notes: str = Field(description="修改建议")


def _fallback_review(draft: str) -> ReviewResult:
    has_citations = "[" in draft and "]" in draft
    enough_length = len(draft) > 400
    approved = has_citations and enough_length
    notes = "结构和引用基本合格。" if approved else "请补充引用、分析深度和结论部分。"
    return ReviewResult(approved=approved, notes=notes)


def review_report(topic: str, draft: str) -> ReviewResult:
    try:
        llm = get_chat_model().with_structured_output(ReviewResult)
        result = llm.invoke(
            [
                (
                    "system",
                    "你是严格但建设性的 Reviewer。请检查研究报告是否围绕主题展开，"
                    "是否有引用、是否结构完整。",
                ),
                ("human", f"研究主题：{topic}\n\n报告草稿：\n{draft}"),
            ]
        )
        return result
    except Exception:
        return _fallback_review(draft)
