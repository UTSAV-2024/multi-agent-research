from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ================================
    # API SETTINGS
    # ================================

    API_TITLE: str = "Multi-Agent Research Platform"

    API_VERSION: str = "1.0.0"

    DEBUG: bool = True

    # ================================
    # LLM SETTINGS
    # ================================

    MODEL_NAME: str = "llama-3.1-8b-instant"

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
    # AUTH SETTINGS
    # ================================

    JWT_SECRET: str = "your-secret-key"

    # ================================
    # VECTOR DB SETTINGS
    # ================================

    VECTOR_DB: str = "chromadb"

    VECTOR_COLLECTION: str = "research_documents"

    # ================================
    # GROQ
    # ================================

    GROQ_API_KEY: str

    class Config:

        env_file = ".env"


settings = Settings()