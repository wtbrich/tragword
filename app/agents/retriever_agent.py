from __future__ import annotations

from app.config import get_settings
from app.rag.retriever import retrieve_snippets
from app.tools.web_search import search_web


def _deduplicate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item.get("source", ""), item.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def gather_evidence(topic: str, sub_questions: list[str]) -> dict[str, list[dict[str, str]]]:
    settings = get_settings()
    evidence: dict[str, list[dict[str, str]]] = {}
    for question in sub_questions:
        local_results = retrieve_snippets(f"{topic}\n{question}", top_k=settings.top_k)
        if settings.search_enabled:
            web_results = search_web(question, max_results=max(2, settings.top_k // 2))
        else:
            web_results = []
        evidence[question] = _deduplicate(local_results + web_results)
    return evidence
