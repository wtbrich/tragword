from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


def build_embeddings(
    provider: str,
    model_name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> object:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "openai":
        kwargs: dict[str, object] = {"model": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_embeddings() -> object:
    settings = get_settings()
    model_name = (
        settings.openai_embedding_model
        if settings.embedding_provider.lower() == "openai"
        else settings.embedding_model
    )
    return build_embeddings(
        settings.embedding_provider,
        model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
