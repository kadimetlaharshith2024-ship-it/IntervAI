from uuid import UUID
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel

class ResumeBase(BaseModel):
    filename: str

class ResumeCreate(ResumeBase):
    raw_text: str

class ResumeResponse(ResumeBase):
    id: UUID
    character_count: int
    created_at: Optional[datetime] = None
    raw_text_preview: str
    detected_sections: Dict[str, bool]

    class Config:
        from_attributes = True