from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    kwargs: dict[str, object] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
    }
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
