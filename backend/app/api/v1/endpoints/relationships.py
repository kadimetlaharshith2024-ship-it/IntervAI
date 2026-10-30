import traceback
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user_optional, verify_resume_access
from app.models.user import User
from app.models.relationship import Relationship
from app.schemas.relationship import RelationshipResponse, RelationshipTriplet
from app.services.relationship_service import extract_and_store_relationships_advanced
from app.services.relationship_baseline import extract_baseline_relationships

router = APIRouter()


@router.post("/{resume_id}/extract-relationships", response_model=RelationshipResponse)
def generate_relationships(
    resume_id: UUID, 
    mode: str = Query("advanced", pattern="^(advanced|baseline)$"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Extracts semantic relationships.
    mode='advanced' executes the contextual, anti-hallucination dependency engine.
    mode='baseline' runs the legacy pairwise co-occurrence extractor for Phase 9 benchmarking.
    """
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    try:
        if mode == "advanced":
            triplets = extract_and_store_relationships_advanced(resume_id, db)
        else:
            triplets = extract_baseline_relationships(resume_id, db)

        return RelationshipResponse(
            resume_id=resume_id,
            extraction_mode=mode,
            total_relationships=len(triplets),
            relationships=triplets
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Relationship extraction failed: {str(exc)}")


@router.get("/{resume_id}", response_model=RelationshipResponse)
def get_relationships(
    resume_id: UUID, 
    mode: str = Query("advanced", pattern="^(advanced|baseline)$"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    try:
        records = db.query(Relationship).filter(
            Relationship.resume_id == resume_id
        ).all()

        if not records:
            raise HTTPException(status_code=404, detail="No relationships found for this resume.")

        triplets = [
            RelationshipTriplet(
                source_entity=r.source_entity,
                relation_type=r.relation_type,
                target_entity=r.target_entity,
                section=r.section,
                context_entity=r.context_entity,
                evidence_text=r.evidence_text,
                confidence=r.confidence,
                extraction_method=r.extraction_method
            )
            for r in records
        ]

        return RelationshipResponse(
            resume_id=resume_id,
            extraction_mode=mode,
            total_relationships=len(triplets),
            relationships=triplets
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch relationships: {str(exc)}")