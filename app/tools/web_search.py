from __future__ import annotations


def _normalize_results(raw: object, query: str) -> list[dict[str, str]]:
    if isinstance(raw, list):
        normalized: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                text = (
                    str(item.get("snippet") or item.get("body") or item.get("content") or "")
                    .strip()
                )
                title = str(item.get("title") or "").strip()
                link = str(item.get("link") or item.get("url") or "").strip()
                combined = " - ".join(part for part in [title, text] if part)
                normalized.append(
                    {
                        "text": combined or title or text or query,
                        "source": link or "duckduckgo",
                    }
                )
            else:
                text = str(item).strip()
                if text:
                    normalized.append({"text": text, "source": "duckduckgo"})
        return normalized

    text = str(raw).strip()
    return [{"text": text or query, "source": "duckduckgo"}] if text else []


def search_web(query: str, *, max_results: int = 3) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except Exception:
        return []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    return _normalize_results(raw_results, query)
