from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-rag"

    # Chunking parameters
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Retrieval parameters
    top_k: int = 5
    relevance_threshold: float = 0.35

    # Gemini embedding model — text-embedding-004 outputs 768 dimensions
    embedding_model: str = "models/text-embedding-004"
    embedding_dim: int = 768

    # LLM via Groq
    llm_model: str = "llama3-8b-8192"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
