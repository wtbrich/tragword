from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, snippets: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    if not snippets or top_k <= 0:
        return []
    try:
        model = get_cross_encoder()
        scores = model.predict([(query, item.get('text', '')) for item in snippets])
        ranked: list[dict[str, str]] = []
        for item, score in zip(snippets, scores, strict=False):
            updated = dict(item)
            updated['score'] = f'{float(score):.6f}'
            ranked.append(updated)
        ranked.sort(key=lambda item: float(item.get('score', '0')), reverse=True)
        return ranked[:top_k]
    except Exception:
        fallback: list[dict[str, str]] = []
        for item in snippets[:top_k]:
            updated = dict(item)
            updated.setdefault('score', '0.000000')
            fallback.append(updated)
        return fallback
