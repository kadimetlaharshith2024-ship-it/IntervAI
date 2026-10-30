import pytest
from app.core.config import Settings

def test_config_development_defaults_allowed():
    """Development environment permits default development secrets."""
    dev_settings = Settings(
        ENVIRONMENT="development",
        JWT_SECRET_KEY="dev-insecure-jwt-secret-key-change-in-production-intervai-auth-2026",
        LLM_PROVIDER="mock",
        LLM_API_KEY=""
    )
    # Should not raise
    dev_settings.validate_startup_configuration()

def test_config_production_rejects_insecure_jwt_secret():
    """Production environment must strictly reject default insecure JWT secret."""
    prod_settings = Settings(
        ENVIRONMENT="production",
        JWT_SECRET_KEY="dev-insecure-jwt-secret-key-change-in-production-intervai-auth-2026",
        LLM_PROVIDER="mock",
        LLM_API_KEY=""
    )
    with pytest.raises(ValueError, match="JWT_SECRET_KEY must be set to a secure"):
        prod_settings.validate_startup_configuration()

def test_config_production_rejects_missing_llm_key():
    """Production environment with gemini or openai requires LLM_API_KEY."""
    prod_settings = Settings(
        ENVIRONMENT="production",
        JWT_SECRET_KEY="a-very-strong-production-secret-key-that-is-not-default-123456",
        LLM_PROVIDER="gemini",
        LLM_API_KEY=""
    )
    with pytest.raises(ValueError, match="LLM_API_KEY must be configured"):
        prod_settings.validate_startup_configuration()

def test_config_production_accepts_valid_config():
    """Production environment with valid secret and API key succeeds."""
    prod_settings = Settings(
        ENVIRONMENT="production",
        JWT_SECRET_KEY="a-very-strong-production-secret-key-that-is-not-default-123456",
        LLM_PROVIDER="gemini",
        LLM_API_KEY="valid-production-api-key"
    )
    prod_settings.validate_startup_configuration()
