import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY
from sqlalchemy.exc import SQLAlchemyError

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.session_service import session_service
from app.services.knowledge_state_service import knowledge_state_service
from app.services.adaptive_selection_service import (
    adaptive_selection_engine,
    AdaptiveSelectionConfig,
)
from app.schemas.adaptive_selection import (
    AdaptiveStrategyType,
    NextQuestionSelectionResult,
    AdaptiveQuestionSelectionRequest,
    AdaptiveNextQuestionResponse,
)
from app.schemas.rag import QuestionGenerationRequest, QuestionGenerationResponse, QuestionOutputSchema
from app.schemas.knowledge_state import ClaimVerificationStatus, TopicCoverageStatus, GapNature
from app.services.rag.rag_pipeline import rag_pipeline
from app.core.database import get_db
from app.main import app


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session: Session):
    """Provides a FastAPI TestClient with database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_resume(db_session: Session) -> Resume:
    """Creates a sample candidate resume with skills, relationships, and claims."""
    resume = Resume(
        id=uuid4(),
        filename="alex_cloud_architect.pdf",
        raw_text="Alex Cloud Architect. Languages: Python, FastAPI, PostgreSQL, Docker, Redis. Built PaymentGateway."
    )
    db_session.add(resume)
    db_session.flush()

    # Skills
    sk1 = Skill(id=401, name="Python", category="Language")
    sk2 = Skill(id=402, name="FastAPI", category="Framework")
    sk3 = Skill(id=403, name="PostgreSQL", category="Database")
    sk4 = Skill(id=404, name="Docker", category="Tool/DevOps")
    sk5 = Skill(id=405, name="Redis", category="Database")
    db_session.add_all([sk1, sk2, sk3, sk4, sk5])
    db_session.flush()

    rsk1 = ResumeSkill(resume_id=resume.id, skill_id=401, extraction_confidence=0.95)
    rsk2 = ResumeSkill(resume_id=resume.id, skill_id=402, extraction_confidence=0.92)
    rsk3 = ResumeSkill(resume_id=resume.id, skill_id=403, extraction_confidence=0.90)
    rsk4 = ResumeSkill(resume_id=resume.id, skill_id=404, extraction_confidence=0.88)
    rsk5 = ResumeSkill(resume_id=resume.id, skill_id=405, extraction_confidence=0.85)
    db_session.add_all([rsk1, rsk2, rsk3, rsk4, rsk5])

    # Relationship
    rel = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="COMMUNICATES_WITH",
        target_entity="PostgreSQL",
        context_entity="PaymentGateway",
        section="Experience",
        evidence_text="Built asynchronous transactional pipeline between FastAPI and PostgreSQL.",
        confidence=0.94
    )
    db_session.add(rel)

    # Claim
    claim = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-REDIS-99",
        section="Experience",
        project="PaymentGateway",
        target_entity="Redis",
        relation_type="DEPLOYED_FOR",
        claim_text="Implemented distributed Redis locking preventing double payments under 10k RPS load.",
        evidence_text="Implemented distributed Redis locking preventing double payments under 10k RPS load."
    )
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(resume)
    return resume


def make_mock_question(
    question_text: str = "How do you design scalable APIs?",
    category: str = "Architecture",
    difficulty: int = 3,
    target_project: str = "PaymentGateway",
    target_entity: str = "FastAPI",
    evidence_ids: list = None,
    evidence_text: str = "Built asynchronous transactional pipeline between FastAPI and PostgreSQL."
) -> QuestionOutputSchema:
    return QuestionOutputSchema(
        question=question_text,
        category=category,
        difficulty=difficulty,
        target_project=target_project,
        target_entity=target_entity,
        evidence_ids=evidence_ids or ["CLM-REDIS-99"],
        evidence_text=evidence_text,
        rationale="Testing architecture and trade-offs.",
        grounding_score=1.0,
        relevance_score=1.0,
        technical_quality_score=1.0,
        trace_id="trc_mock_test"
    )


# =========================================================================
# TEST 1: Unsupported Question Targets Validation & Safe Rejection
# =========================================================================
def test_unsupported_question_target_rejected(db_session: Session, test_resume: Resume):
    """
    Step 2: Validates that question targets not supported by the candidate's resume
    (unsupported claim, project, relationship) are cleanly rejected.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # 1. Unsupported claim ID
    req_unsupported_claim = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        session_id=session.id,
        target_claim_id="CLM-NONEXISTENT-999"
    )
    resp1 = rag_pipeline.generate_grounded_question(test_resume.id, req_unsupported_claim, db_session)
    assert resp1.validation_status == "REJECTED"
    assert "not supported by candidate resume" in resp1.message

    # 2. Unsupported project
    req_unsupported_proj = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        session_id=session.id,
        target_project="HyperdriveBlockchainQuantum"
    )
    resp2 = rag_pipeline.generate_grounded_question(test_resume.id, req_unsupported_proj, db_session)
    assert resp2.validation_status == "REJECTED"
    assert "not supported by candidate resume" in resp2.message

    # 3. Unsupported relationship
    req_unsupported_rel = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        session_id=session.id,
        target_relationship="Kafka STREAMS_TO Cassandra"
    )
    resp3 = rag_pipeline.generate_grounded_question(test_resume.id, req_unsupported_rel, db_session)
    assert resp3.validation_status == "REJECTED"
    assert "not supported by candidate resume" in resp3.message


# =========================================================================
# TEST 2: Duplicate Question Prevention & Anti-Duplication Prompt Guidance
# =========================================================================
def test_duplicate_question_prevention(db_session: Session, test_resume: Resume):
    """
    Step 2: Validates that questions closely repeating an already-asked question
    are identified as semantic duplicates by validator and rejected with retry guidance.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Insert an existing question into the session
    existing_q = InterviewQuestion(
        resume_id=test_resume.id,
        session_id=session.id,
        question_text="How did you configure Redis distributed locking in PaymentGateway to handle high concurrency?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis",
        evidence_text="Implemented distributed Redis locking preventing double payments under 10k RPS load."
    )
    db_session.add(existing_q)
    db_session.commit()

    from app.services.rag.validator import grounding_validator
    evidence_str = "In PaymentGateway, implemented distributed Redis locking preventing double payments under 10k RPS load."
    candidate_dup = make_mock_question(
        question_text="How did you configure Redis distributed locking in PaymentGateway to handle high concurrency?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis",
        evidence_ids=["CLM-REDIS-99"],
        evidence_text=evidence_str
    )

    # 1. Direct semantic duplicate check via embedding cosine similarity
    is_dup_direct, msg_direct = grounding_validator.check_semantic_duplicate(
        candidate_dup.question,
        [existing_q]
    )
    assert is_dup_direct
    assert "semantically duplicate" in msg_direct.lower()

    # 2. End-to-end validator check
    resolved_items = {
        "CLM-REDIS-99": {
            "evidence_id": "CLM-REDIS-99",
            "text": evidence_str,
            "source_type": "claim"
        }
    }

    is_valid, reason, details = grounding_validator.validate(
        question=candidate_dup,
        allowed_projects=["PaymentGateway"],
        allowed_entities=["Redis"],
        resolved_evidence_items=resolved_items,
        existing_questions=[existing_q],
        available_claims={"CLM-REDIS-99": existing_q},
        available_relationships={}
    )

    assert not is_valid
    assert details.get("code") == "semantic_duplicate"


# =========================================================================
# TEST 3: Invalid and Ended Sessions Cannot Generate Questions
# =========================================================================
def test_invalid_and_ended_sessions_rejected(db_session: Session, test_resume: Resume, client: TestClient):
    """
    Step 3: Verifies that sessions that have ended or completed cannot generate
    new interview questions or preview questions (HTTP 400).
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)
    session_service.end_interview(session.id, db_session, reason="USER_COMPLETED")

    # 1. Attempting next-question generation on ended session raises 400
    with pytest.raises(HTTPException) as exc1:
        adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)
    assert exc1.value.status_code == 400
    assert "already ENDED" in exc1.value.detail or "already COMPLETED" in exc1.value.detail

    # 2. Attempting preview on ended session raises 400
    with pytest.raises(HTTPException) as exc2:
        adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    assert exc2.value.status_code == 400

    # 3. API endpoint for next-question returns 400
    resp_post = client.post(f"/api/v1/sessions/{session.id}/next-question")
    assert resp_post.status_code == 400

    # 4. API endpoint for preview returns 400
    resp_get = client.get(f"/api/v1/sessions/{session.id}/next-question/preview")
    assert resp_get.status_code == 400


# =========================================================================
# TEST 4: Maximum Interview-Question Budget Enforcement
# =========================================================================
def test_maximum_interview_question_budget_enforcement(db_session: Session, test_resume: Resume):
    """
    Step 3: Verifies that when a session reaches its configured maximum question budget,
    select_and_generate_next_question returns BUDGET_EXHAUSTED and preview raises HTTP 400.
    """
    session = session_service.create_session(
        resume_id=test_resume.id,
        db=db_session,
        session_metadata={"adaptive_config": {"max_questions": 2}}
    )

    # Insert 2 questions to reach budget
    for i in range(2):
        q = InterviewQuestion(
            resume_id=test_resume.id,
            session_id=session.id,
            question_text=f"Question {i + 1} on architecture?",
            category="Architecture",
            difficulty=3
        )
        db_session.add(q)
    db_session.commit()

    # Generation returns explicit BUDGET_EXHAUSTED
    resp = adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)
    assert resp.status == "BUDGET_EXHAUSTED"
    assert "maximum question budget" in resp.message
    assert resp.selection is None

    # Preview raises 400 indicating budget reached
    with pytest.raises(HTTPException) as exc:
        adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    assert exc.value.status_code == 400
    assert "maximum question budget" in exc.value.detail


# =========================================================================
# TEST 5: Per-Topic and Consecutive Follow-Up Budgets
# =========================================================================
def test_per_topic_and_consecutive_followup_budgets(db_session: Session, test_resume: Resume):
    """
    Step 3: Verifies that consecutive follow-ups and per-topic limits are strictly
    enforced, rotating to breadth coverage (UNCOVERED_TOPIC) when limits are hit.
    """
    session = session_service.create_session(
        resume_id=test_resume.id,
        db=db_session,
        session_metadata={"adaptive_config": {"max_consecutive_followups": 1, "max_questions_per_topic": 2}}
    )

    # Add parent question and follow-up question (depth = 1 follow-up)
    q1 = InterviewQuestion(
        resume_id=test_resume.id,
        session_id=session.id,
        question_text="Tell me about FastAPI routing.",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    db_session.add(q1)
    db_session.flush()

    q2 = InterviewQuestion(
        resume_id=test_resume.id,
        session_id=session.id,
        parent_question_id=q1.id,
        question_text="How does FastAPI handle dependency injection in routers?",
        category="Implementation",
        difficulty=3,
        target_entity="FastAPI"
    )
    db_session.add(q2)
    db_session.flush()

    # Add evaluated answer to q2 with partial score
    ans2 = InterviewAnswer(
        session_id=session.id,
        question_id=q2.id,
        candidate_answer="FastAPI uses Depends for DI.",
        status="EVALUATED",
        overall_score=60.0,
        missing_concepts=["yield dependencies", "sub-dependencies"],
        evaluation_confidence=0.85
    )
    db_session.add(ans2)
    db_session.commit()

    # Next target selection should NOT allow another follow-up because max_consecutive_followups = 1
    # It must rotate to an uncovered topic
    target = adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    assert not target.is_follow_up
    assert target.parent_question_id is None
    assert target.strategy == AdaptiveStrategyType.UNCOVERED_TOPIC


# =========================================================================
# TEST 6: Preview Requests Do Not Mutate Session State
# =========================================================================
def test_preview_requests_do_not_mutate_state(db_session: Session, test_resume: Resume, client: TestClient):
    """
    Step 3: Verifies preview requests deterministically preview targets
    without persisting questions, mutating difficulty, or changing session counters.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session, initial_difficulty=3)
    initial_total_q = session.total_questions
    initial_diff = session.current_difficulty

    # Call preview multiple times
    for _ in range(3):
        resp = client.get(f"/api/v1/sessions/{session.id}/next-question/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategy" in data
        assert "target_topic" in data

    # Verify state remains completely unmutated
    db_session.refresh(session)
    assert session.total_questions == initial_total_q
    assert session.current_difficulty == initial_diff
    assert db_session.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).count() == 0


# =========================================================================
# TEST 7: RAG Generation Failure Handled Gracefully
# =========================================================================
def test_rag_generation_failure_handled_gracefully(db_session: Session, test_resume: Resume):
    """
    Step 4: Verifies that if RAG generation fails (e.g. INSUFFICIENT_EVIDENCE),
    the selection engine returns EVIDENCE_UNAVAILABLE without corrupting session counters.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req_failed",
        trace_id="trc_failed",
        question=None,
        validation_status="INSUFFICIENT_EVIDENCE",
        message="No evidence found for selected topic.",
        attempts_count=3,
        latencies_ms={}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)

    assert resp.status == "EVIDENCE_UNAVAILABLE"
    assert "Could not generate grounded question" in resp.message
    db_session.refresh(session)
    assert session.total_questions == 0


# =========================================================================
# TEST 8: Database Persistence Failure Rollback
# =========================================================================
def test_database_persistence_failure_rollback(db_session: Session, test_resume: Resume):
    """
    Step 4: Simulates a database commit failure during session state update,
    ensuring rollback is executed and HTTP 500 is cleanly raised.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    mock_q = make_mock_question(
        question_text="How does Python manage memory?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Python"
    )
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req_success",
        trace_id="trc_success",
        question_id=uuid4(),
        question=mock_q,
        validation_status="ACCEPTED",
        message="Success",
        attempts_count=1,
        latencies_ms={}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        with patch.object(db_session, "commit", side_effect=SQLAlchemyError("Connection terminated")):
            with pytest.raises(HTTPException) as exc:
                adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)
            assert exc.value.status_code == 500
            assert "Database persistence failure" in exc.value.detail


# =========================================================================
# TEST 9: Session Recovery After Reload
# =========================================================================
def test_session_recovery_after_reload(db_session: Session, test_resume: Resume):
    """
    Step 4: Verifies that after generating a question, full session reload via
    get_interview_state accurately reflects the new question in UNANSWERED state.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    mock_q = make_mock_question(
        question_text="How does PostgreSQL handle multi-version concurrency control?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="PostgreSQL"
    )
    q_id = uuid4()
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req_mvcc",
        trace_id="trc_mvcc",
        question_id=q_id,
        question=mock_q,
        validation_status="ACCEPTED",
        message="Generated",
        attempts_count=1,
        latencies_ms={}
    )

    # Persist question in DB
    new_q = InterviewQuestion(
        id=q_id,
        resume_id=test_resume.id,
        session_id=session.id,
        question_text=mock_q.question,
        category=mock_q.category,
        difficulty=mock_q.difficulty,
        target_project=mock_q.target_project,
        target_entity=mock_q.target_entity,
        evidence_ids=mock_q.evidence_ids,
        rationale=mock_q.rationale
    )
    db_session.add(new_q)
    db_session.commit()

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)
        assert resp.status == "GENERATED"

    # Reload session state from scratch
    reloaded_state = session_service.get_interview_state(session.id, db_session)
    assert len(reloaded_state.asked_questions) >= 1
    assert reloaded_state.asked_questions[-1].status == "UNANSWERED"
    assert reloaded_state.progress.completed_questions == 0
    assert reloaded_state.progress.total_questions >= 1


# =========================================================================
# TEST 10: Repeated Next-Question Request Idempotency
# =========================================================================
def test_request_idempotency_key_prevents_duplicate_generation(db_session: Session, test_resume: Resume):
    """
    Step 4: Verifies that passing request_idempotency_key on retried requests
    returns the identical cached response without creating duplicate questions.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)
    idem_key = f"idem_{uuid4().hex}"

    mock_q = make_mock_question(
        question_text="Explain Redis caching strategies in PaymentGateway.",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis"
    )
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req_idem_1",
        trace_id="trc_idem_1",
        question_id=uuid4(),
        question=mock_q,
        validation_status="ACCEPTED",
        message="Generated",
        attempts_count=1,
        latencies_ms={}
    )

    req_override = AdaptiveQuestionSelectionRequest(request_idempotency_key=idem_key)

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        # First request
        resp1 = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session,
            request_override=req_override
        )
        assert resp1.status == "GENERATED"
        first_q_id = resp1.question_id

        # Second request with identical idempotency key (simulated retry)
        resp2 = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session,
            request_override=req_override
        )
        assert resp2.status == "GENERATED"
        assert resp2.question_id == first_q_id
        assert resp2.question_text == resp1.question_text

    # Verify session total_questions did not double
    db_session.refresh(session)
    assert session.total_questions == 1


# =========================================================================
# TEST 11: Missing or Inconsistent Knowledge State Recovery
# =========================================================================
def test_missing_or_inconsistent_knowledge_state_recovery(db_session: Session, test_resume: Resume):
    """
    Step 4: Verifies that if session_state["knowledge_state"] is missing or
    corrupted (e.g. topics is None or non-dict), get_or_initialize_knowledge_state
    detects corruption and safely re-initializes from resume without crashing.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Deliberately corrupt knowledge state
    bad_state = dict(session.session_state or {})
    bad_state["knowledge_state"] = {
        "topics": None,  # Corrupt
        "projects": "invalid_string",  # Corrupt
        "relationships": None,
        "claims": []
    }
    session.session_state = bad_state
    db_session.commit()
    db_session.refresh(session)

    # Calling get_or_initialize_knowledge_state must heal the state
    recovered_k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    assert isinstance(recovered_k_state, dict)
    assert isinstance(recovered_k_state["topics"], dict)
    assert "Python" in recovered_k_state["topics"]
    assert isinstance(recovered_k_state["projects"], dict)
    assert isinstance(recovered_k_state["relationships"], list)


# =========================================================================
# TEST 12: Skip, Timeout, Refusal, and Invalid-Score Handling
# =========================================================================
def test_skip_timeout_refusal_and_invalid_score_handling(db_session: Session, test_resume: Resume):
    """
    Step 5: Verifies:
    1. Unassessed topics are NOT classified as demonstrated weaknesses.
    2. Skipped/failed/unanswered questions do not receive fabricated technical scores.
    3. Failed evaluations do not corrupt knowledge state.
    4. Difficulty stays within configured bounds.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)
    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)

    # 1. Check unassessed topics in summary: must NOT be in demonstrated_weaknesses
    summary = k_state.get("summary", {})
    assert len(summary.get("demonstrated_weaknesses", [])) == 0
    assert "Python" in summary.get("unassessed_topics", [])

    # 2. Insert question and a FAILED answer attempt without score
    q = InterviewQuestion(
        resume_id=test_resume.id,
        session_id=session.id,
        question_text="Describe Docker containerization workflow.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    db_session.add(q)
    db_session.flush()

    failed_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        candidate_answer="My answer timed out or failed.",
        status="FAILED",
        overall_score=None,  # No score
        evaluation_confidence=0.0
    )
    db_session.add(failed_ans)
    db_session.commit()

    # update_from_evaluation must gracefully skip without modifying mastery or fabricating scores
    k_state_after = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=failed_ans,
        db=db_session
    )
    docker_topic = k_state_after["topics"].get("Docker", {})
    assert docker_topic.get("answers_evaluated_count", 0) == 0
    assert docker_topic.get("latest_score") is None
    assert docker_topic.get("mastery_score") == 0.0

    # 3. Refusal answer handling: Candidate refused -> difficulty de-escalates or maintains min_difficulty
    refusal_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        candidate_answer="I don't know, skip this question.",
        status="EVALUATED",
        overall_score=0.0,
        evaluator_model="fast_path_refusal",
        evaluation_feedback="Candidate refused question",
        evaluation_confidence=1.0
    )
    db_session.add(refusal_ans)
    db_session.commit()

    # Adaptation check: target difficulty should not drop below min_difficulty (1)
    session.current_difficulty = 1
    db_session.commit()
    target_sel = adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    assert target_sel.difficulty >= 1


# =========================================================================
# TEST 13: Full Compatibility with Phases 7.1–7.5
# =========================================================================
def test_phase7_compatibility_end_to_end_lifecycle(db_session: Session, test_resume: Resume):
    """
    Step 6 & 7: Verifies end-to-end integration across all Phases 7.1-7.6:
    Session creation -> Deterministic Preview -> Adaptive Generation ->
    Answer submission & evaluation -> Scorecard calculation -> Knowledge State sync.
    """
    # 1. Create session (Phase 7.1)
    session = session_service.create_session(resume_id=test_resume.id, db=db_session, initial_difficulty=3)
    assert session.status == "IN_PROGRESS"

    # 2. Preview deterministic target (Phase 7.3 preview)
    preview = session_service.preview_next_question_selection(session.id, db_session)
    assert preview.strategy in AdaptiveStrategyType.__members__.values()
    assert preview.difficulty == 3

    # 3. Generate adaptive next question (Phase 7.3 execution)
    mock_q = make_mock_question(
        question_text="How do you design scalable APIs using FastAPI?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="FastAPI"
    )
    q_id = uuid4()
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req_compat",
        trace_id="trc_compat",
        question_id=q_id,
        question=mock_q,
        validation_status="ACCEPTED",
        message="Question successfully generated.",
        attempts_count=1,
        latencies_ms={}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        gen_resp = session_service.select_and_generate_next_question(session.id, db_session)
        assert gen_resp.status == "GENERATED"
        assert gen_resp.question_id is not None

    # Persist question record matching mock question_id
    q_row = InterviewQuestion(
        id=gen_resp.question_id,
        resume_id=test_resume.id,
        session_id=session.id,
        question_text=mock_q.question,
        category=mock_q.category,
        difficulty=mock_q.difficulty,
        target_project=mock_q.target_project,
        target_entity=mock_q.target_entity,
        evidence_ids=mock_q.evidence_ids,
        rationale=mock_q.rationale
    )
    db_session.add(q_row)
    db_session.commit()

    # 4. Submit evaluated answer (Phase 7.2 & 7.4)
    ans = InterviewAnswer(
        session_id=session.id,
        question_id=gen_resp.question_id,
        candidate_answer="FastAPI uses Starlette and Pydantic with async handlers for high throughput.",
        status="EVALUATED",
        overall_score=85.0,
        evaluation_confidence=0.90,
        evaluation_feedback="Strong technical explanation with clear understanding of async event loop."
    )
    db_session.add(ans)
    db_session.commit()

    # Update knowledge state from evaluation (Phase 7.2)
    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=gen_resp.question_id,
        answer=ans,
        db=db_session
    )

    # Verify score adaptation (Phase 7.4): Next target difficulty should escalate to 4
    next_target = session_service.preview_next_question_selection(session.id, db_session)
    assert next_target.difficulty == 4
    assert next_target.strategy in [AdaptiveStrategyType.DIFFICULTY_ESCALATION, AdaptiveStrategyType.UNCOVERED_TOPIC]

    # Verify interview state serialization (Phase 7.1)
    state = session_service.get_interview_state(session.id, db_session)
    assert state.progress.completed_questions == 1
    assert "FastAPI" in state.tested_technologies
