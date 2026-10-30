from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "IntervAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/intervai"

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_MAX_RETRIES: int = 3

    # Hybrid Retrieval Settings
    RAG_TOP_K_VECTOR: int = 5
    RAG_TOP_K_GRAPH: int = 8
    RAG_TOP_K_LEXICAL: int = 5
    RAG_TOP_K_FINAL: int = 6
    RAG_CONTEXT_TOKEN_BUDGET: int = 2048

    # Quality & Adaptive Thresholds (Part B7)
    RAG_DUPLICATE_SIMILARITY_THRESHOLD: float = 0.82
    RAG_GROUNDING_SCORE_MINIMUM: float = 0.65
    RAG_CLAIM_SIMILARITY_MINIMUM: float = 0.35
    ADAPTIVE_ROLLING_WINDOW_SIZE: int = 3
    ADAPTIVE_CLAIM_QUOTA_PERCENT: float = 0.20
    ADAPTIVE_PROJECT_QUOTA_PERCENT: float = 0.20
    ADAPTIVE_HIGH_SCORE_THRESHOLD: float = 75.0
    ADAPTIVE_LOW_SCORE_THRESHOLD: float = 50.0
    ADAPTIVE_MAX_CANDIDATE_RETRIES: int = 2
    ADAPTIVE_CANDIDATE_TIMEOUT_SECONDS: float = 12.0

    # Latency Budget & Retry Limits (Task 5)
    TOTAL_QUESTION_DEADLINE_SECONDS: float = 12.0
    MIN_LLM_CALL_BUDGET: float = 2.5
    MAX_CANDIDATE_ATTEMPTS: int = 2
    MAX_INTERNAL_RAG_ATTEMPTS: int = 2

    # Feature Flags (Part C)
    PREFETCH_ENABLED: bool = False

    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "openai", "mock"] = Field(
        default="gemini",
        description="Active LLM provider. 'mock' is restricted to automated test suites."
    )
    LLM_MODEL_NAME: str = "gemini-3.1-flash-lite"
    LLM_API_KEY: str = Field(default="", description="API key for the selected provider")
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_OUTPUT_TOKENS: int = 2048
    LLM_MAX_RETRIES: int = 3

    # Dedicated Model & Token Controls (Part A6)
    GENERATION_MODEL: Optional[str] = Field(
        default=None,
        description="Dedicated model name for question generation. If unset, falls back to LLM_MODEL_NAME."
    )
    EVALUATION_MODEL: Optional[str] = Field(
        default=None,
        description="Dedicated model name for answer evaluation. If unset, falls back to LLM_MODEL_NAME."
    )
    GENERATION_MAX_OUTPUT_TOKENS: Optional[int] = Field(
        default=None,
        description="Max output tokens for question generation (typically lower). If unset, falls back to LLM_MAX_OUTPUT_TOKENS."
    )
    EVALUATION_MAX_OUTPUT_TOKENS: Optional[int] = Field(
        default=None,
        description="Max output tokens for answer evaluation. If unset, falls back to LLM_MAX_OUTPUT_TOKENS."
    )
    # Optional retrieval top-k overrides (aliases for RAG_TOP_K_*)
    RAG_VECTOR_TOP_K: Optional[int] = None
    RAG_BM25_TOP_K: Optional[int] = None
    RAG_FINAL_TOP_K: Optional[int] = None

    # Environment
    ENVIRONMENT: Literal["development", "test", "production"] = "development"

    # Auth & JWT Settings
    JWT_SECRET_KEY: str = Field(
        default="dev-insecure-jwt-secret-key-change-in-production-intervai-auth-2026",
        description="Secret key for JWT encoding and signature verification"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # CORS Allowed Origins
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

    # Rate Limiting Settings (Phase 7.3)
    RATE_LIMITING_ENABLED: bool = True
    RATE_LIMIT_NEXT_QUESTION_PER_MIN: int = 20
    RATE_LIMIT_EVALUATION_PER_MIN: int = 25
    RATE_LIMIT_RESUME_UPLOAD_PER_MIN: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    def validate_startup_configuration(self) -> None:
        """
        Validates critical configuration at application startup.
        Fails safely if production configuration uses default secrets or missing provider keys.
        """
        if self.ENVIRONMENT == "production":
            insecure_secret = "dev-insecure-jwt-secret-key-change-in-production-intervai-auth-2026"
            if not self.JWT_SECRET_KEY or self.JWT_SECRET_KEY == insecure_secret:
                raise ValueError(
                    "Production configuration error: JWT_SECRET_KEY must be set to a secure, "
                    "non-default secret in production environments."
                )
            if self.LLM_PROVIDER in ["gemini", "openai"] and not self.LLM_API_KEY:
                raise ValueError(
                    f"Production configuration error: LLM_API_KEY must be configured "
                    f"for provider '{self.LLM_PROVIDER}' in production."
                )

settings = Settings()