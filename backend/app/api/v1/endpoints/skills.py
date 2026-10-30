from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user_optional, verify_resume_access
from app.models.user import User
from app.models.skill import Skill, ResumeSkill
from app.schemas.skill import EntityExtractionResponse, ExtractedEntity
from app.services.entity_service import extract_and_store_resume_entities

router = APIRouter()


@router.post("/{resume_id}/extract-entities", response_model=EntityExtractionResponse)
def extract_entities(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    entities = extract_and_store_resume_entities(resume_id, db)
    return EntityExtractionResponse(
        resume_id=resume_id,
        total_extracted=len(entities),
        entities=entities
    )


@router.get("/{resume_id}", response_model=EntityExtractionResponse)
def get_resume_entities(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves previously extracted skills and entities for a resume.
    """
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    links = (
        db.query(ResumeSkill)
        .join(Skill, ResumeSkill.skill_id == Skill.id)
        .filter(ResumeSkill.resume_id == resume_id)
        .all()
    )
    entities = [
        ExtractedEntity(
            name=link.skill.name,
            category=link.skill.category or "Other",
            confidence=link.extraction_confidence
        )
        for link in links
    ]
    return EntityExtractionResponse(
        resume_id=resume_id,
        total_extracted=len(entities),
        entities=entities
    )