from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user_optional, verify_resume_access
from app.models.user import User
from app.models.rag import InterviewQuestion, RetrievalTrace, ResumeClaim
from app.schemas.rag import (
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    ResumeIndexResponse,
    CandidateClaimSchema
)
from app.services.rag.rag_pipeline import rag_pipeline

router = APIRouter()


@router.post("/{resume_id}/index", response_model=ResumeIndexResponse)
def index_resume(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Indexes chunks, generates embeddings, and extracts grounded atomic claims."""
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    return rag_pipeline.index_resume(resume_id, db)


@router.post("/generate/{resume_id}", response_model=QuestionGenerationResponse)
def generate_question(
    resume_id: UUID,
    request: QuestionGenerationRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Executes Hybrid RAG to produce a grounded interview question with full provenance."""
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    return rag_pipeline.generate_grounded_question(resume_id, request, db)


@router.get("/questions/{resume_id}")
def get_resume_questions(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Retrieves all generated interview questions for a candidate."""
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    questions = db.query(InterviewQuestion).filter(InterviewQuestion.resume_id == resume_id).all()
    return {
        "resume_id": resume_id,
        "total_questions": len(questions),
        "questions": [
            {
                "id": q.id,
                "question": q.question_text,
                "category": q.category,
                "difficulty": q.difficulty,
                "target_project": q.target_project,
                "target_entity": q.target_entity,
                "evidence_text": q.evidence_text,
                "grounding_score": q.grounding_score,
                "trace_id": q.retrieval_trace_id,
                "created_at": q.created_at
            }
            for q in questions
        ]
    }


@router.get("/claims/{resume_id}", response_model=List[CandidateClaimSchema])
def get_resume_claims(
    resume_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Retrieves all extracted atomic claims for a candidate."""
    verify_resume_access(resume_id=resume_id, db=db, current_user=current_user)
    return db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).all()


@router.get("/traces/{trace_id}")
def get_retrieval_trace(
    trace_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Retrieves end-to-end RAG retrieval and validation telemetry."""
    trace = db.query(RetrievalTrace).filter(RetrievalTrace.trace_id == trace_id).first()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace record not found.")
    if trace.resume_id:
        verify_resume_access(resume_id=trace.resume_id, db=db, current_user=current_user)
    return trace