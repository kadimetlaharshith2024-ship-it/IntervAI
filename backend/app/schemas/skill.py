from uuid import UUID
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class ExtractedEntity(BaseModel):
    name: str
    category: str  # 'Language', 'Framework', 'Database', 'Tool/DevOps', 'Core Concept'
    confidence: float

class EntityExtractionResponse(BaseModel):
    resume_id: UUID
    total_extracted: int
    entities: List[ExtractedEntity]

    model_config = ConfigDict(from_attributes=True)