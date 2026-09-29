from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-rag"

    # Chunking parameters
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Retrieval parameters
    top_k: int = 5
    relevance_threshold: float = 0.35

    # gemini-embedding-2 outputs 3072 dimensions
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 3072

    # LLM via Groq
    llm_model: str = "openai/gpt-oss-20b"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
