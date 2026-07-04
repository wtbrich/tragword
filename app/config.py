from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    embedding_provider: str = Field(default="huggingface", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        alias="EMBEDDING_MODEL",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    milvus_uri: str = Field(default="./.milvus/milvus_local.db", alias="MILVUS_DB_URI")
    milvus_collection_name: str = Field(default="tragword", alias="MILVUS_COLLECTION_NAME")
    top_k: int = Field(default=4, alias="TOP_K")
    hybrid_enabled: bool = Field(default=True, alias="HYBRID_ENABLED")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")
    rerank_candidates: int = Field(default=10, alias="RERANK_CANDIDATES")
    multiquery_enabled: bool = Field(default=False, alias="MULTIQUERY_ENABLED")
    search_enabled: bool = Field(default=True, alias="SEARCH_ENABLED")
    max_revisions: int = Field(default=2, alias="MAX_REVISIONS")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
