from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ================================
    # API SETTINGS
    # ================================

    API_TITLE: str = "Multi-Agent Research Platform"

    API_VERSION: str = "1.0.0"

    DEBUG: bool = True

    # ================================
    # LLM (Groq)
    # ================================

    LLM_PROVIDER: str = "groq"

    MODEL_NAME: str = "llama-3.1-8b-instant"

    GROQ_API_KEY: str

    TEMPERATURE: float = 0.1

    DEFAULT_MAX_TOKENS: int = 1000

    SUMMARY_MAX_TOKENS: int = 300

    FACTCHECK_MAX_TOKENS: int = 500

    REPORT_MAX_TOKENS: int = 1200

    # ================================
    # RETRIEVAL SETTINGS
    # ================================

    MAX_SEARCH_RESULTS: int = 5

    MAX_ARTICLE_LENGTH: int = 3000

    REQUEST_TIMEOUT: int = 10

    MAX_RETRIES: int = 3

    # ================================
    # LOGGING SETTINGS
    # ================================

    LOG_LEVEL: str = "INFO"

    # ================================
    # ENVIRONMENT SETTINGS
    # ================================

    ENVIRONMENT: str = "development"

    # ================================
    # DATABASE SETTINGS
    # ================================

    MONGODB_URL: str

    DATABASE_NAME: str

    # ================================
    # HYBRID RETRIEVAL SETTINGS
    # ================================

    HYBRID_SEMANTIC_WEIGHT: float = 0.7

    HYBRID_KEYWORD_WEIGHT: float = 0.3

    HYBRID_RETRIEVAL_MULTIPLIER: int = 3

    # ================================
    # EVIDENCE RETRIEVAL SETTINGS
    # ================================

    MAX_CHUNKS_PER_SOURCE: int = 2

    # ================================
    # AUTH SETTINGS
    # ================================

    JWT_SECRET: str = "your-secret-key"

    # ================================
    # VECTOR DB SETTINGS
    # ================================

    VECTOR_ENABLED: bool = True

    VECTOR_COLLECTION: str = "research_chunks"

    CHROMA_PERSIST_DIR: str = "./chroma_data"

    # ================================
    # EMBEDDING SETTINGS
    # ================================

    EMBEDDINGS_ENABLED: bool = True

    EMBEDDING_PROVIDER: str = "sentence_transformers"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    EMBEDDING_BATCH_SIZE: int = 32

    class Config:

        env_file = ".env"


settings = Settings()