from __future__ import annotations

import os
from collections.abc import Callable

from app.config import get_settings


def _ensure_download_timeout() -> None:
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')


def warmup_embeddings() -> None:
    _ensure_download_timeout()
    from app.rag.embeddings import get_embeddings

    embeddings = get_embeddings()
    embeddings.embed_query('warmup')


def warmup_reranker() -> None:
    _ensure_download_timeout()
    from app.rag.rerank import get_cross_encoder

    reranker = get_cross_encoder()
    reranker.predict([('q', 'd')])


def warmup_models(
    progress: Callable[[str, str], None] | None = None,
) -> None:
    if progress is not None:
        progress('embedding', 'start')
    warmup_embeddings()
    if progress is not None:
        progress('embedding', 'done')
    if get_settings().rerank_enabled:
        if progress is not None:
            progress('reranker', 'start')
        warmup_reranker()
        if progress is not None:
            progress('reranker', 'done')
    elif progress is not None:
        progress('reranker', 'skipped')
