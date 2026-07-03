from __future__ import annotations

from app.llm import get_chat_model


def _format_evidence(retrieved: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []
    ref_index = 1
    for question, items in retrieved.items():
        lines.append(f"### {question}")
        if not items:
            lines.append("- 暂无资料")
            continue
        for item in items:
            text = item.get("text", "").strip()
            source = item.get("source", "")
            lines.append(f"- [{ref_index}] {text}（来源：{source}）")
            ref_index += 1
    return "\n".join(lines)


def _fallback_draft(
    topic: str,
    sub_questions: list[str],
    retrieved: dict[str, list[dict[str, str]]],
    review_notes: str,
) -> str:
    references: list[str] = []
    body: list[str] = [
        f"# {topic}",
        "",
        "## 研究结论",
        "本次研究整合了本地知识库与可选联网搜索结果。",
    ]
    if review_notes:
        body.extend(["", "## 上一轮审稿意见", review_notes])

    body.extend(["", "## 子问题拆解"])
    for question in sub_questions:
        body.append(f"- {question}")

    body.extend(["", "## 资料摘录"])
    ref_no = 1
    for question, items in retrieved.items():
        body.append(f"### {question}")
        for item in items:
            text = item.get("text", "").strip()
            source = item.get("source", "")
            body.append(f"- 证据[{ref_no}]：{text}")
            references.append(f"[{ref_no}] {source}")
            ref_no += 1

    if references:
        body.extend(["", "## 参考来源", *references])
    return "\n".join(body).strip() + "\n"


def write_report(
    topic: str,
    sub_questions: list[str],
    retrieved: dict[str, list[dict[str, str]]],
    review_notes: str = "",
) -> str:
    evidence_text = _format_evidence(retrieved)
    prompt = [
        (
            "system",
            "你是研究报告 Writer。请根据给定资料撰写 Markdown 报告，"
            "要求结构清晰、尽量使用引用编号。",
        ),
        (
            "human",
            f"研究主题：{topic}\n\n"
            f"子问题：\n- "
            + "\n- ".join(sub_questions)
            + "\n\n"
            f"审稿意见：{review_notes or '无'}\n\n"
            f"资料：\n{evidence_text}\n\n"
            "请输出完整 Markdown 报告，包含摘要、分析、结论和参考来源。",
        ),
    ]
    try:
        response = get_chat_model().invoke(prompt)
        content = getattr(response, "content", "").strip()
        return content or _fallback_draft(topic, sub_questions, retrieved, review_notes)
    except Exception:
        return _fallback_draft(topic, sub_questions, retrieved, review_notes)
