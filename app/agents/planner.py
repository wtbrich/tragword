from pydantic import BaseModel, Field

from app.llm import get_chat_model


class PlanResult(BaseModel):
    sub_questions: list[str] = Field(min_length=1)


def _fallback_plan(topic: str) -> list[str]:
    base = topic.strip() or "该主题"
    return [
        f"{base} 的背景和定义是什么？",
        f"{base} 需要哪些关键技术和流程？",
        f"{base} 的实践难点和最佳实践是什么？",
    ]


def plan_sub_questions(topic: str) -> list[str]:
    try:
        llm = get_chat_model().with_structured_output(PlanResult)
        result = llm.invoke(
            [
                (
                    "system",
                    "你是研究助手的 Planner。请把用户主题拆成 3 到 5 个清晰、可检索的子问题。",
                ),
                ("human", f"研究主题：{topic}"),
            ]
        )
        questions = [item.strip() for item in result.sub_questions if item.strip()]
        return questions or _fallback_plan(topic)
    except Exception:
        return _fallback_plan(topic)
