from typing import Optional
from app.core.config import settings
from app.services.llm.base import BaseLLMProvider
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.mock import MockProvider

def get_llm_provider(
    force_mock: bool = False,
    role: Optional[str] = None,
    model_name: Optional[str] = None,
    max_output_tokens: Optional[int] = None
) -> BaseLLMProvider:
    if force_mock or settings.LLM_PROVIDER == "mock":
        return MockProvider()

    provider_name = settings.LLM_PROVIDER.lower()

    # Determine resolved model name and max tokens based on role (Part A6)
    if role == "generation":
        resolved_model = model_name or settings.GENERATION_MODEL or settings.LLM_MODEL_NAME
        resolved_tokens = max_output_tokens or settings.GENERATION_MAX_OUTPUT_TOKENS or getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 600)
    elif role == "evaluation":
        resolved_model = model_name or settings.EVALUATION_MODEL or settings.LLM_MODEL_NAME
        resolved_tokens = max_output_tokens or settings.EVALUATION_MAX_OUTPUT_TOKENS or getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 2048)
    else:
        resolved_model = model_name or settings.LLM_MODEL_NAME
        resolved_tokens = max_output_tokens or getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 600)

    if provider_name == "gemini":
        if not settings.LLM_API_KEY:
            raise ValueError(
                "CRITICAL: LLM_PROVIDER is configured as 'gemini' but LLM_API_KEY is unset in .env. "
                "Provide a valid API key or set LLM_PROVIDER='mock' for local unit testing."
            )
        return GeminiProvider(
            api_key=settings.LLM_API_KEY,
            model_name=resolved_model,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_output_tokens=resolved_tokens
        )

    if provider_name == "openai":
        if not settings.LLM_API_KEY:
            raise ValueError(
                "CRITICAL: LLM_PROVIDER is configured as 'openai' but LLM_API_KEY is unset in .env."
            )
        return OpenAIProvider(
            api_key=settings.LLM_API_KEY,
            model_name=resolved_model,
            timeout=settings.LLM_TIMEOUT_SECONDS
        )

    raise ValueError(f"Unknown LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Must be 'gemini', 'openai', or 'mock'.")