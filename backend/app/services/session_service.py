import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from uuid import UUID, uuid4
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.resume import Resume
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.schemas.session import (
    AskedQuestionState,
    KnowledgeGapItem,
    InterviewProgressState,
    InterviewStateResponse,
)
from app.services.evaluation.scorecard import scorecard_aggregator

logger = logging.getLogger(__name__)


class InterviewSessionService:
    """
    Manages the production lifecycle of interview sessions, question associations,
    session state tracking, difficulty progression, and state persistence.
    """

    def create_session(
        self,
        resume_id: UUID,
        db: Session,
        initial_difficulty: int = 3,
        session_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[UUID] = None
    ) -> InterviewSession:
        """
        Creates a new interview session for an existing candidate resume.
        """
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume record not found.")

        # Ensure session cannot be started with another user's resume
        if resume.user_id is not None:
            if user_id is not None and resume.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: Cannot start interview with another user's resume."
                )
            if user_id is None:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required to start interview with this resume."
                )

        resolved_user_id = user_id or resume.user_id

        if initial_difficulty < 1 or initial_difficulty > 5:
            raise HTTPException(status_code=400, detail="initial_difficulty must be between 1 and 5.")

        now_utc = datetime.now(timezone.utc)
        initial_state = {
            "tested_skills": [],
            "tested_technologies": [],
            "tested_relationships": [],
            "tested_claims": [],
            "tested_projects": [],
            "knowledge_gaps": [],
            "metadata": session_metadata or {},
        }

        session = InterviewSession(
            id=uuid4(),
            user_id=resolved_user_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=initial_difficulty,
            completed_questions=0,
            total_questions=0,
            category_scores={},
            session_state=initial_state,
            started_at=now_utc,
            completed_at=None,
        )
        db.add(session)
        db.flush()

        try:
            from app.services.knowledge_state_service import knowledge_state_service
            from sqlalchemy.orm.attributes import flag_modified
            k_state = knowledge_state_service.initialize_knowledge_state(session, db)
            initial_state["knowledge_state"] = k_state
            session.session_state = initial_state
            flag_modified(session, "session_state")
        except Exception as e:
            logger.warning(f"Failed to pre-initialize knowledge state for session {session.id}: {e}")

        db.commit()
        db.refresh(session)

        # TASK 5: Pre-generate Question 1 immediately upon session creation
        try:
            from app.services.adaptive_selection_service import adaptive_selection_service
            v_str = adaptive_selection_service._get_state_version(session, db)
            adaptive_selection_service.execute_prefetch(session.id, v_str, db)
            db.refresh(session)
        except Exception as e:
            logger.warning(f"Failed to pre-generate Question 1 for session {session.id}: {e}")

        logger.info(f"Created interview session {session.id} for resume {resume_id} (difficulty={initial_difficulty})")
        return session

    def start_interview(
        self,
        db: Session,
        resume_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        initial_difficulty: int = 3,
        session_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[UUID] = None
    ) -> InterviewSession:
        """
        Starts or resumes an interview session.
        Idempotent: If an active session already exists, returns it without creating duplicates.
        """
        if session_id is not None:
            session = self.get_session(session_id, db)
            if session.user_id is not None:
                if user_id is None:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required to access this interview session."
                    )
                if session.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: You do not have permission to access this session."
                    )
            elif session.resume and session.resume.user_id is not None:
                if user_id is None:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required to access this interview session."
                    )
                if session.resume.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: You do not have permission to access this session."
                    )

            # If session was ENDED or COMPLETED, restarting reopens it to IN_PROGRESS
            if session.status in ["ENDED", "COMPLETED"]:
                session.status = "IN_PROGRESS"
                session.completed_at = None
                if initial_difficulty:
                    session.current_difficulty = initial_difficulty
                db.commit()
                db.refresh(session)
            return session

        if resume_id is not None:
            resume = db.query(Resume).filter(Resume.id == resume_id).first()
            if not resume:
                raise HTTPException(status_code=404, detail="Resume record not found.")

            if resume.user_id is not None:
                if user_id is None:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required to start interview with this resume."
                    )
                if resume.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: Cannot start interview with another user's resume."
                    )

            # Check if an active IN_PROGRESS session already exists for this resume
            active_session = (
                db.query(InterviewSession)
                .filter(
                    InterviewSession.resume_id == resume_id,
                    InterviewSession.status == "IN_PROGRESS"
                )
                .order_by(InterviewSession.started_at.desc())
                .first()
            )
            if active_session:
                logger.info(f"Returning existing active interview session {active_session.id} for resume {resume_id}")
                return active_session

            return self.create_session(
                resume_id=resume_id,
                db=db,
                initial_difficulty=initial_difficulty,
                session_metadata=session_metadata,
                user_id=user_id
            )

        raise HTTPException(
            status_code=400,
            detail="Either resume_id or session_id must be provided to start an interview."
        )

    def end_interview(
        self,
        session_id: UUID,
        db: Session,
        reason: str = "USER_COMPLETED",
        feedback_notes: Optional[str] = None
    ) -> InterviewSession:
        """
        Ends an interview session, calculates final scorecard aggregation,
        and transitions the status to COMPLETED (if all questions evaluated) or ENDED.
        Idempotent: Repeated end calls return the already ended session safely.
        """
        session = self.get_session(session_id, db)

        # If already concluded, return safely (idempotent)
        if session.status in ["ENDED", "COMPLETED"]:
            return session

        # Calculate scorecard on all answers submitted so far
        try:
            scorecard_aggregator.calculate_session_scorecard(db, session_id)
            db.refresh(session)
        except Exception as e:
            logger.warning(f"Error calculating scorecard during session {session_id} conclusion: {e}")

        # Transition status
        now_utc = datetime.now(timezone.utc)
        if session.total_questions > 0 and session.completed_questions == session.total_questions:
            session.status = "COMPLETED"
        else:
            session.status = "ENDED"

        session.completed_at = now_utc

        # Persist conclusion metadata in session_state
        state = dict(session.session_state or {})
        state["end_reason"] = reason
        if feedback_notes:
            state["feedback_notes"] = feedback_notes
        session.session_state = state

        # Sync full state
        self.sync_session_state(session, db)

        db.commit()
        db.refresh(session)
        logger.info(f"Interview session {session_id} concluded with status '{session.status}' (reason={reason})")
        return session

    def get_session(self, session_id: UUID, db: Session) -> InterviewSession:
        """
        Retrieves interview session by ID. Raises 404 if not found.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found.")
        return session

    def associate_question(
        self,
        session_id: UUID,
        question_id: UUID,
        db: Session
    ) -> InterviewQuestion:
        """
        Associates an interview question with the specified session.
        Validates resume alignment and prevents reassignment across different sessions.
        """
        session = self.get_session(session_id, db)

        question = db.query(InterviewQuestion).filter(InterviewQuestion.id == question_id).first()
        if not question:
            raise HTTPException(status_code=404, detail="Interview question not found.")

        if question.resume_id != session.resume_id:
            raise HTTPException(
                status_code=400,
                detail="Question does not belong to the candidate resume linked to this session."
            )

        if question.session_id is not None and question.session_id != session_id:
            raise HTTPException(
                status_code=400,
                detail=f"Question is already assigned to another interview session ({question.session_id})."
            )

        if question.session_id != session_id:
            question.session_id = session_id
            db.commit()
            db.refresh(question)

            # If session was completed or ended, adding a new question reopens it to IN_PROGRESS
            if session.status in ["COMPLETED", "ENDED"]:
                session.status = "IN_PROGRESS"
                session.completed_at = None

            try:
                scorecard_aggregator.calculate_session_scorecard(db, session_id)
                self.sync_session_state(session, db)
                db.refresh(session)
            except Exception as e:
                logger.warning(f"Failed to auto-update scorecard/state for session {session_id}: {e}")

        return question

    def get_session_questions(self, session_id: UUID, db: Session) -> List[InterviewQuestion]:
        """
        Lists all interview questions linked to the specified session in chronological order.
        """
        self.get_session(session_id, db)
        return (
            db.query(InterviewQuestion)
            .filter(InterviewQuestion.session_id == session_id)
            .order_by(InterviewQuestion.created_at.asc())
            .all()
        )

    def update_difficulty(self, session_id: UUID, new_difficulty: int, db: Session) -> InterviewSession:
        """
        Updates the adaptive difficulty level for the interview session (1 to 5).
        """
        if new_difficulty < 1 or new_difficulty > 5:
            raise HTTPException(status_code=400, detail="Difficulty level must be between 1 and 5.")

        session = self.get_session(session_id, db)
        session.current_difficulty = new_difficulty
        db.commit()
        db.refresh(session)
        logger.info(f"Updated difficulty for session {session_id} to {new_difficulty}")
        return session

    def get_interview_state(self, session_id: UUID, db: Session) -> InterviewStateResponse:
        """
        Builds the comprehensive interview state response, including asked questions,
        answer attempt integration, tested skills/technologies/relationships/claims,
        knowledge gaps, and overall progress. Synchronizes state to PostgreSQL.
        """
        session = self.get_session(session_id, db)

        # Ensure active in-progress session without explicit end_reason is not prematurely marked COMPLETED
        end_reason = (session.session_state or {}).get("end_reason")
        if session.status == "COMPLETED" and not end_reason:
            session.status = "IN_PROGRESS"
            session.completed_at = None
            db.commit()
            db.refresh(session)

        # 1. Retrieve all questions and answers for this session
        questions = (
            db.query(InterviewQuestion)
            .filter(InterviewQuestion.session_id == session_id)
            .order_by(InterviewQuestion.created_at.asc())
            .all()
        )
        answers = (
            db.query(InterviewAnswer)
            .filter(InterviewAnswer.session_id == session_id)
            .all()
        )

        # 2. Group answers by question_id
        answers_by_question: Dict[UUID, List[InterviewAnswer]] = defaultdict(list)
        for ans in answers:
            answers_by_question[ans.question_id].append(ans)

        # 3. Build AskedQuestionState list & identify latest attempts
        asked_questions: List[AskedQuestionState] = []
        latest_evaluated_answers: Dict[UUID, InterviewAnswer] = {}
        knowledge_gaps: List[KnowledgeGapItem] = []

        tested_skills_set: Set[str] = set()
        tested_tech_set: Set[str] = set()
        tested_rel_set: Set[str] = set()
        tested_claim_set: Set[str] = set()
        tested_proj_set: Set[str] = set()

        for q in questions:
            # Track tested topics from question metadata
            if q.target_entity:
                clean_ent = q.target_entity.strip()
                tested_tech_set.add(clean_ent)
                tested_skills_set.add(clean_ent)
            if q.target_project:
                tested_proj_set.add(q.target_project.strip())
            if q.target_relationship:
                tested_rel_set.add(q.target_relationship.strip())
            if q.targeted_claim:
                tested_claim_set.add(q.targeted_claim.strip())
            if q.expected_concepts:
                for c in q.expected_concepts:
                    if isinstance(c, str) and c.strip():
                        tested_skills_set.add(c.strip())

            q_answers = answers_by_question.get(q.id, [])
            attempt_count = len(q_answers)
            status = "UNANSWERED"
            latest_score = None
            latest_feedback = None

            if q_answers:
                sorted_ans = sorted(
                    q_answers,
                    key=lambda a: (
                        getattr(a, "attempt_number", 0) or 0,
                        a.created_at or datetime.min,
                        str(a.id)
                    ),
                    reverse=True
                )
                latest_ans = sorted_ans[0]
                status = latest_ans.status
                latest_score = latest_ans.overall_score
                latest_feedback = latest_ans.evaluation_feedback

                if latest_ans.status == "EVALUATED":
                    latest_evaluated_answers[q.id] = latest_ans

                    # Accumulate knowledge gaps
                    missing = list(latest_ans.missing_concepts or [])
                    weaknesses = list(latest_ans.weaknesses or [])
                    score = latest_ans.overall_score or 0.0

                    if score < 50.0:
                        severity = "HIGH"
                    elif score < 75.0:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    NON_SUBSTANTIVE_CONCEPTS = {
                        "none", "none significant", "n/a", "na", "no major omissions",
                        "nothing substantive", "none noted", "nil", "nothing", "no missing concepts",
                        "not applicable", "no critical omissions", "no significant omissions",
                        "no omissions", "no major gaps", "none observed"
                    }
                    substantive_missing = [
                        c for c in missing
                        if isinstance(c, str)
                        and c.strip().lower() not in NON_SUBSTANTIVE_CONCEPTS
                        and not any(c.strip().lower().startswith(p) for p in ["none ", "none.", "no major", "no significant", "no critical", "nothing "])
                    ]

                    if score < 75.0 or (score < 85.0 and len(substantive_missing) >= 1):
                        topic = q.target_entity or q.target_project or q.category
                        knowledge_gaps.append(
                            KnowledgeGapItem(
                                question_id=q.id,
                                category=q.category,
                                topic=topic,
                                missing_concepts=substantive_missing,
                                weaknesses=weaknesses,
                                severity=severity,
                                score=score
                            )
                        )

            asked_questions.append(
                AskedQuestionState(
                    question_id=q.id,
                    question_text=q.question_text,
                    category=q.category,
                    difficulty=q.difficulty,
                    target_project=q.target_project,
                    target_entity=q.target_entity,
                    target_relationship=q.target_relationship,
                    targeted_claim=q.targeted_claim,
                    expected_concepts=list(q.expected_concepts or []),
                    status=status,
                    attempt_count=attempt_count,
                    latest_score=latest_score,
                    latest_feedback_summary=latest_feedback,
                    is_follow_up=bool(q.parent_question_id),
                    parent_question_id=q.parent_question_id,
                    asked_at=q.created_at,
                )
            )

        # 4. Progress calculations
        total_q = len(questions)
        completed_q = len(latest_evaluated_answers)
        progress_pct = round((completed_q / total_q) * 100, 1) if total_q > 0 else 0.0

        now_utc = datetime.now(timezone.utc)
        duration_sec = None
        if session.started_at:
            end_t = session.completed_at or now_utc
            try:
                # Handle timezone-aware / naive calculations safely
                start_t = session.started_at if session.started_at.tzinfo else session.started_at.replace(tzinfo=timezone.utc)
                end_t = end_t if end_t.tzinfo else end_t.replace(tzinfo=timezone.utc)
                duration_sec = max(0.0, round((end_t - start_t).total_seconds(), 1))
            except Exception:
                duration_sec = None

        progress = InterviewProgressState(
            completed_questions=completed_q,
            total_questions=total_q,
            progress_percentage=progress_pct,
            overall_score=session.overall_score,
            category_scores=dict(session.category_scores or {}),
            status=session.status,
            duration_seconds=duration_sec,
        )

        # 5. Synchronize state to session_state JSON in PostgreSQL
        state_dict = dict(session.session_state or {})
        state_dict["tested_skills"] = sorted(list(tested_skills_set))
        state_dict["tested_technologies"] = sorted(list(tested_tech_set))
        state_dict["tested_relationships"] = sorted(list(tested_rel_set))
        state_dict["tested_claims"] = sorted(list(tested_claim_set))
        state_dict["tested_projects"] = sorted(list(tested_proj_set))
        state_dict["knowledge_gaps"] = [g.model_dump(mode="json") for g in knowledge_gaps]
        session.session_state = state_dict

        # Keep session counters aligned
        session.total_questions = total_q
        session.completed_questions = completed_q

        db.commit()
        db.refresh(session)

        return InterviewStateResponse(
            session_id=session.id,
            resume_id=session.resume_id,
            status=session.status,
            current_difficulty=session.current_difficulty,
            started_at=session.started_at,
            completed_at=session.completed_at,
            progress=progress,
            asked_questions=asked_questions,
            tested_skills=sorted(list(tested_skills_set)),
            tested_technologies=sorted(list(tested_tech_set)),
            tested_relationships=sorted(list(tested_rel_set)),
            tested_claims=sorted(list(tested_claim_set)),
            tested_projects=sorted(list(tested_proj_set)),
            knowledge_gaps=knowledge_gaps,
            knowledge_state=state_dict.get("knowledge_state"),
            session_metadata=state_dict.get("metadata", {}),
        )

    def sync_session_state(self, session: InterviewSession, db: Session) -> None:
        """
        Lightweight synchronization of tested topics and knowledge gaps into session_state.
        """
        try:
            questions = (
                db.query(InterviewQuestion)
                .filter(InterviewQuestion.session_id == session.id)
                .all()
            )
            tested_skills: Set[str] = set()
            tested_tech: Set[str] = set()
            tested_rel: Set[str] = set()
            tested_claims: Set[str] = set()
            tested_proj: Set[str] = set()

            for q in questions:
                if q.target_entity:
                    clean = q.target_entity.strip()
                    tested_tech.add(clean)
                    tested_skills.add(clean)
                if q.target_project:
                    tested_proj.add(q.target_project.strip())
                if q.target_relationship:
                    tested_rel.add(q.target_relationship.strip())
                if q.targeted_claim:
                    tested_claims.add(q.targeted_claim.strip())
                if q.expected_concepts:
                    for c in q.expected_concepts:
                        if isinstance(c, str) and c.strip():
                            tested_skills.add(c.strip())

            state = dict(session.session_state or {})
            state["tested_skills"] = sorted(list(tested_skills))
            state["tested_technologies"] = sorted(list(tested_tech))
            state["tested_relationships"] = sorted(list(tested_rel))
            state["tested_claims"] = sorted(list(tested_claims))
            state["tested_projects"] = sorted(list(tested_proj))
            session.session_state = state
        except Exception as e:
            logger.warning(f"Error during lightweight sync_session_state for session {session.id}: {e}")

    def get_candidate_knowledge_state(self, session_id: UUID, db: Session):
        """
        Retrieves the structured Candidate Knowledge State for a session.
        """
        from app.services.knowledge_state_service import knowledge_state_service
        return knowledge_state_service.format_knowledge_state_response(session_id=session_id, db=db)

    def reconcile_candidate_knowledge_state(self, session_id: UUID, db: Session):
        """
        Reconciles the Candidate Knowledge State by replaying all historical answers.
        """
        from app.services.knowledge_state_service import knowledge_state_service
        knowledge_state_service.reconcile_knowledge_state(session_id=session_id, db=db)
        return knowledge_state_service.format_knowledge_state_response(session_id=session_id, db=db)

    def select_and_generate_next_question(
        self,
        session_id: UUID,
        db: Session,
        request_override=None,
        background_tasks=None
    ):
        """
        Selects next adaptive question target and executes grounded RAG generation (Phase 7.3).
        """
        from app.services.adaptive_selection_service import adaptive_selection_engine
        return adaptive_selection_engine.select_and_generate_next_question(
            session_id=session_id,
            db=db,
            request_override=request_override,
            background_tasks=background_tasks
        )

    def preview_next_question_selection(
        self,
        session_id: UUID,
        db: Session,
        request_override=None
    ):
        """
        Previews deterministic next question selection target and rationale without generating or persisting.
        """
        from app.services.adaptive_selection_service import adaptive_selection_engine
        return adaptive_selection_engine.determine_next_question_target(
            session_id=session_id,
            db=db,
            request_override=request_override
        )


session_service = InterviewSessionService()


