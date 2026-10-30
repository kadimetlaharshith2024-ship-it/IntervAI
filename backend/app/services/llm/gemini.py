import time
import json
import re
from typing import Dict, Any
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult

class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite",
        timeout: float = 30.0,
        max_output_tokens: int = 2048
    ):
        if not api_key:
            raise ValueError(
                "CRITICAL: GeminiProvider initialized without an API key. "
                "Set LLM_API_KEY in backend/.env before running in production."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
        except ImportError:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai.GenerativeModel(self.model_name)
                self._sdk_type = "legacy"
            except ImportError:
                raise RuntimeError(
                    "No Google GenAI SDK installed. Run: pip install google-genai"
                )

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.2
    ) -> LLMGenerationResult:
        t0 = time.time()

        actual_model = self.model_name
        if self._sdk_type == "genai":
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
                max_output_tokens=self.max_output_tokens
            )
            candidate_models = [self.model_name]
            for m in ["gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-flash-lite-latest"]:
                if m not in candidate_models:
                    candidate_models.append(m)

            response = None
            last_err = None
            for mod in candidate_models:
                for retry in range(2):
                    try:
                        response = self._client.models.generate_content(
                            model=mod,
                            contents=prompt,
                            config=config
                        )
                        actual_model = mod
                        break
                    except Exception as e:
                        last_err = e
                        err_str = str(e)
                        if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                            time.sleep(0.8)
                            continue
                        raise
                if response is not None:
                    break

            if response is None:
                raise last_err

            raw_text = response.text
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "candidates_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                }
        else:
            full_prompt = f"System Instructions:\n{system_instruction}\n\nTask:\n{prompt}"
            response = self._client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "max_output_tokens": self.max_output_tokens
                }
            )
            raw_text = response.text
            usage = None

        latency_ms = round((time.time() - t0) * 1000, 2)
        parsed = self._extract_json(raw_text)

        return LLMGenerationResult(
            raw_content=raw_text,
            parsed_json=parsed,
            provider="gemini",
            model=actual_model,
            latency_ms=latency_ms,
            usage=usage
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cleaned)
            return json.loads(sanitized)