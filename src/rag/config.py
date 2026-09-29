"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-rag"

    # Chunking parameters
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Retrieval parameters
    top_k: int = 5
    relevance_threshold: float = 0.35

    # Embedding model and its output dimension
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # LLM
    llm_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
