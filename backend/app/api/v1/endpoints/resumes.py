from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.database import get_db
from app.core.deps import get_current_user, get_current_user_optional, verify_resume_access
from app.models.user import User
from app.models.resume import Resume
from app.schemas.resume import ResumeResponse
from app.schemas.session import SessionResponse
from app.services.session_service import session_service
from app.services.resume_service import process_and_store_resume, detect_sections

router = APIRouter()


@router.get("", response_model=List[ResumeResponse])
@router.get("/", response_model=List[ResumeResponse], include_in_schema=False)
def list_user_resumes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Lists all resumes belonging to the currently authenticated user.
    """
    resumes = (
        db.query(Resume)
        .filter(Resume.user_id == current_user.id)
        .order_by(Resume.created_at.desc())
        .all()
    )
    return [
        ResumeResponse(
            id=r.id,
            filename=r.filename,
            character_count=len(r.raw_text),
            created_at=r.created_at,
            raw_text_preview=r.raw_text[:400],
            detected_sections=detect_sections(r.raw_text)
        )
        for r in resumes
    ]


@router.post(
    "/upload",
    response_model=ResumeResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_RESUME_UPLOAD_PER_MIN, 60))]
)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF resume, parses sections, and associates ownership if the user is authenticated.
    Eagerly runs idempotent entity and relationship extraction.
    """
    user_id = current_user.id if current_user else None
    resume_record, detected_sections = await process_and_store_resume(file, db, user_id=user_id)

    # Eager idempotent extraction for immediate frontend readiness
    try:
        from app.services.entity_service import extract_and_store_resume_entities
        from app.services.relationship_service import extract_and_store_relationships_advanced
        extract_and_store_resume_entities(resume_record.id, db)
        extract_and_store_relationships_advanced(resume_record.id, db)
    except Exception:
        # Non-blocking: if background extraction encounters an issue, the resume upload itself succeeds
        pass

    return ResumeResponse(
        id=resume_record.id,
        filename=resume_record.filename,
        character_count=len(resume_record.raw_text),
        created_at=resume_record.created_at,
        raw_text_preview=resume_record.raw_text[:400],
        detected_sections=detected_sections
    )


@router.get("/{resume_id}", response_model=ResumeResponse)
def get_resume(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves a resume by ID after verifying access permissions.
    """
    resume = verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)

    return ResumeResponse(
        id=resume.id,
        filename=resume.filename,
        character_count=len(resume.raw_text),
        created_at=resume.created_at,
        raw_text_preview=resume.raw_text[:400],
        detected_sections=detect_sections(resume.raw_text)
    )


@router.post("/{resume_id}/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session_for_resume(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Convenience endpoint to initialize an interview session directly for a resume.
    Enforces that the user owns the resume before session creation.
    """
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    user_id = current_user.id if current_user else None
    return session_service.create_session(resume_id=resume_id, db=db, user_id=user_id)