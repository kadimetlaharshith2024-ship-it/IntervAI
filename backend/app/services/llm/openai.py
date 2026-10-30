import time
import json
from typing import Dict, Any
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", timeout: float = 30.0):
        if not api_key:
            raise ValueError(
                "CRITICAL: OpenAIProvider initialized without an API key. Set LLM_API_KEY in backend/.env."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise RuntimeError("openai library is not installed. Run: pip install openai")

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.2
    ) -> LLMGenerationResult:
        t0 = time.time()
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
        )
        latency_ms = round((time.time() - t0) * 1000, 2)
        raw_text = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_text)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "candidates_tokens": response.usage.completion_tokens
            }

        return LLMGenerationResult(
            raw_content=raw_text,
            parsed_json=parsed,
            provider="openai",
            model=self.model_name,
            latency_ms=latency_ms,
            usage=usage
        )