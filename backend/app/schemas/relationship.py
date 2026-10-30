from uuid import UUID
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

class RelationshipTriplet(BaseModel):
    source_entity: str
    relation_type: str
    target_entity: str
    section: str
    context_entity: Optional[str] = None
    evidence_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_method: str = "dependency_parsed"

    model_config = ConfigDict(from_attributes=True)

class RelationshipResponse(BaseModel):
    resume_id: UUID
    extraction_mode: str
    total_relationships: int
    relationships: List[RelationshipTriplet]

    model_config = ConfigDict(from_attributes=True)

class ExtractionQuery(BaseModel):
    mode: str = Field(default="advanced", description="'advanced' (contextual NLP) or 'baseline' (rule-based co-occurrence)")