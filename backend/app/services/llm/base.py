from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class LLMGenerationResult(BaseModel):
    raw_content: str
    parsed_json: Dict[str, Any]
    provider: str
    model: str
    latency_ms: float
    usage: Optional[Dict[str, int]] = None

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.2
    ) -> LLMGenerationResult:
        """Generates a structured JSON response from the LLM provider."""
        pass