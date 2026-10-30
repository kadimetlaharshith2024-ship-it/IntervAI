import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime

from app.services.adaptive_selection_service import (
    AdaptiveSelectionEngine,
    AdaptiveSelectionConfig,
    AdaptiveStrategyType,
    NextQuestionSelectionResult,
    adaptive_selection_engine,
)
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.schemas.rag import QuestionOutputSchema, QuestionGenerationResponse
from app.schemas.adaptive_selection import AdaptiveNextQuestionResponse


class TestPartCPrefetching:
    """Unit tests for Part C: Background pre-generation, caching, and invalidation rules."""

    def test_prefetch_cached_and_consumed_on_valid_breadth_strategy(self):
        """When prefetch is valid, it is consumed, topic is updated, and prefetch cache is cleared."""
        resume_id = uuid4()
        session_id = uuid4()
        q_id = uuid4()

        session = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            total_questions=1,
            session_state={
                "metadata": {"prefetch_enabled": True},
                "knowledge_state": {
                    "topics": {
                        "FastAPI": {"questions_asked_count": 1, "coverage_status": "QUESTION_ASKED"},
                        "Docker": {"questions_asked_count": 0, "coverage_status": "NOT_CONSIDERED"}
                    },
                    "projects": {},
                    "relationships": [],
                    "claims": [],
                    "knowledge_gaps": [],
                    "summary": {},
                    "coverage": {}
                },
                "prefetched_question": {
                    "question_id": str(q_id),
                    "question": {
                        "question": "How do you containerize a multi-stage FastAPI build in Docker?",
                        "category": "Implementation",
                        "difficulty": 3,
                        "grounding_score": 0.95,
                        "relevance_score": 0.95,
                        "technical_quality_score": 0.95,
                        "trace_id": "trace-docker",
                        "evidence_ids": ["chk_docker"],
                        "evidence_text": "Docker multi-stage build.",
                        "rationale": "Testing Docker breadth"
                    },
                    "selection": {
                        "strategy": "UNCOVERED_TOPIC",
                        "target_topic": "Docker",
                        "target_entity": "Docker",
                        "target_project": None,
                        "target_claim_id": None,
                        "target_relationship": None,
                        "category": "Implementation",
                        "difficulty": 3,
                        "rationale": "Selecting Docker for breadth.",
                        "is_follow_up": False,
                        "parent_question_id": None,
                        "consecutive_followup_count": 0
                    },
                    "knowledge_state_version": "q:1_ans:0_eval:0_diff:3",
                    "difficulty": 3,
                    "strategy": "UNCOVERED_TOPIC",
                    "created_at": datetime.utcnow().isoformat(),
                    "consumed": False
                }
            }
        )

        db = MagicMock()
        db.bind.dialect.name = "postgresql"
        db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = session
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.side_effect = [
            1,  # existing questions count
            1,  # q_cnt for state version
            0,  # ans_cnt for state version
            0,  # eval_cnt for state version
            1,  # q_cnt for next prefetch check
            0,  # ans_cnt
            0,  # eval_cnt
        ]
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        engine = AdaptiveSelectionEngine()

        # Call select_and_generate_next_question
        with patch.object(engine, "determine_next_question_target") as mock_target:
            mock_target.return_value = NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.UNCOVERED_TOPIC,
                target_topic="Docker",
                target_entity="Docker",
                category="Implementation",
                difficulty=3,
                rationale="Breadth"
            )

            resp = engine.select_and_generate_next_question(session_id, db)

            assert resp.status == "GENERATED"
            assert resp.question_id == q_id
            assert "Docker" in resp.question_text
            assert resp.latencies_ms.get("prefetch_hit") == 1.0
            assert resp.latencies_ms.get("rag_ms") == 0.0

            # Verified that prefetched_question was popped from session_state (consumed at most once!)
            assert "prefetched_question" not in session.session_state
            assert session.session_state["_prefetch_metrics"]["prefetches_consumed"] == 1

    def test_prefetch_invalidated_when_answer_evaluated_and_state_version_changes(self):
        """When an answer was evaluated, state version changed -> stale prefetch is discarded and generated live."""
        resume_id = uuid4()
        session_id = uuid4()
        old_q_id = uuid4()
        live_q_id = uuid4()

        session = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            total_questions=1,
            session_state={
                "metadata": {"prefetch_enabled": True},
                "knowledge_state": {
                    "topics": {
                        "FastAPI": {"questions_asked_count": 1, "coverage_status": "QUESTION_ASKED"},
                        "Docker": {"questions_asked_count": 0, "coverage_status": "NOT_CONSIDERED"}
                    },
                    "projects": {},
                    "relationships": [],
                    "claims": [],
                    "knowledge_gaps": [],
                    "summary": {},
                    "coverage": {}
                },
                "prefetched_question": {
                    "question_id": str(old_q_id),
                    "question": {"question": "Old prefetched question"},
                    "selection": {"strategy": "UNCOVERED_TOPIC", "target_topic": "Docker"},
                    "knowledge_state_version": "q:1_ans:0_eval:0_diff:3",  # Stale version!
                    "difficulty": 3,
                    "strategy": "UNCOVERED_TOPIC",
                    "created_at": datetime.utcnow().isoformat(),
                    "consumed": False
                }
            }
        )

        db = MagicMock()
        db.bind.dialect.name = "postgresql"
        db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = session
        db.query.return_value.filter.return_value.first.return_value = session
        # 1 evaluated answer exists! Version becomes q:1_ans:1_eval:1_diff:3 != q:1_ans:0_eval:0_diff:3
        db.query.return_value.filter.return_value.count.return_value = 1
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        engine = AdaptiveSelectionEngine()

        live_output = QuestionOutputSchema(
            question="Live generated fresh question?",
            category="Implementation",
            difficulty=3,
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="live-trace",
            evidence_ids=["ev_1"],
            evidence_text="Fresh evidence",
            rationale="Fresh live rationale"
        )
        live_resp = QuestionGenerationResponse(
            resume_id=resume_id,
            request_id="req-live",
            trace_id="live-trace",
            question_id=live_q_id,
            question=live_output,
            validation_status="ACCEPTED",
            message="Live generated",
            attempts_count=1,
            latencies_ms={"total_ms": 100.0}
        )

        with patch("app.services.adaptive_selection_service.rag_pipeline.generate_grounded_question", return_value=live_resp):
            resp = engine.select_and_generate_next_question(session_id, db)

            assert resp.status == "GENERATED"
            assert resp.question_id == live_q_id
            assert resp.question_text == "Live generated fresh question?"
            # Stale prefetch was recorded as discarded!
            assert session.session_state["_prefetch_metrics"]["prefetches_discarded"] == 1
            assert "prefetched_question" not in session.session_state

    def test_followup_strategy_never_serves_prefetch(self):
        """Follow-up strategies (gap, contradiction, partial clarification, escalation) are ALWAYS generated live."""
        resume_id = uuid4()
        session_id = uuid4()
        live_q_id = uuid4()

        session = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            total_questions=1,
            session_state={
                "metadata": {"prefetch_enabled": True},
                "knowledge_state": {
                    "topics": {"FastAPI": {"questions_asked_count": 1, "coverage_status": "QUESTION_ASKED"}},
                    "projects": {},
                    "relationships": [],
                    "claims": [],
                    "knowledge_gaps": [{"gap_id": "g1", "topic": "FastAPI", "is_addressed": False, "severity": "HIGH"}],
                    "summary": {},
                    "coverage": {}
                },
                "prefetched_question": {
                    "question_id": str(uuid4()),
                    "question": {"question": "Prefetched Docker question"},
                    "selection": {"strategy": "UNCOVERED_TOPIC", "target_topic": "Docker"},
                    "knowledge_state_version": "q:1_ans:0_eval:0_diff:3",
                    "difficulty": 3,
                    "strategy": "UNCOVERED_TOPIC",
                    "created_at": datetime.utcnow().isoformat(),
                    "consumed": False
                }
            }
        )

        db = MagicMock()
        db.bind.dialect.name = "postgresql"
        db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = session
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.side_effect = [
            1, 1, 0, 0, 1, 0, 0
        ]
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        engine = AdaptiveSelectionEngine()

        # Suppose next target selection is a follow-up (GAP_FOLLOW_UP)
        with patch.object(engine, "determine_next_question_target") as mock_target:
            mock_target.return_value = NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.GAP_FOLLOW_UP,
                target_topic="FastAPI",
                target_entity="FastAPI",
                category="Fundamentals",
                difficulty=3,
                rationale="Probing demonstrated gap",
                is_follow_up=True
            )

            live_output = QuestionOutputSchema(
                question="Live gap follow-up on FastAPI?",
                category="Fundamentals",
                difficulty=3,
                grounding_score=0.9,
                relevance_score=0.9,
                technical_quality_score=0.9,
                trace_id="gap-trace",
                evidence_ids=["ev_1"],
                evidence_text="FastAPI gap",
                rationale="Gap follow-up"
            )
            live_resp = QuestionGenerationResponse(
                resume_id=resume_id,
                request_id="req-gap",
                trace_id="gap-trace",
                question_id=live_q_id,
                question=live_output,
                validation_status="ACCEPTED",
                message="Gap live generated",
                attempts_count=1,
                latencies_ms={"total_ms": 90.0}
            )

            with patch("app.services.adaptive_selection_service.rag_pipeline.generate_grounded_question", return_value=live_resp):
                resp = engine.select_and_generate_next_question(session_id, db)

                assert resp.status == "GENERATED"
                assert resp.selection.strategy == AdaptiveStrategyType.GAP_FOLLOW_UP
                assert resp.question_text == "Live gap follow-up on FastAPI?"
                # Prefetch was discarded because strategy changed to follow-up!
                assert session.session_state["_prefetch_metrics"]["prefetches_discarded"] == 1

    def test_stale_prefetch_difficulty_mismatch_discarded(self):
        """When session difficulty changed (e.g. escalated 3 -> 4), prefetch at diff 3 is discarded."""
        resume_id = uuid4()
        session_id = uuid4()
        live_q_id = uuid4()

        session = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=4,  # Escalated to 4!
            total_questions=1,
            session_state={
                "metadata": {"prefetch_enabled": True},
                "knowledge_state": {
                    "topics": {"FastAPI": {"questions_asked_count": 1, "coverage_status": "QUESTION_ASKED"}},
                    "projects": {},
                    "relationships": [],
                    "claims": [],
                    "knowledge_gaps": [],
                    "summary": {},
                    "coverage": {}
                },
                "prefetched_question": {
                    "question_id": str(uuid4()),
                    "question": {"question": "Prefetched Docker question at diff 3"},
                    "selection": {"strategy": "UNCOVERED_TOPIC", "target_topic": "Docker"},
                    "knowledge_state_version": "q:1_ans:0_eval:0_diff:3",
                    "difficulty": 3,  # Cached at diff 3!
                    "strategy": "UNCOVERED_TOPIC",
                    "created_at": datetime.utcnow().isoformat(),
                    "consumed": False
                }
            }
        )

        db = MagicMock()
        db.bind.dialect.name = "postgresql"
        db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = session
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.return_value = 1
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        engine = AdaptiveSelectionEngine()

        live_output = QuestionOutputSchema(
            question="Fresh difficulty 4 question?",
            category="Architecture",
            difficulty=4,
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="diff4-trace",
            evidence_ids=["ev_1"],
            evidence_text="Architecture evidence",
            rationale="Difficulty 4 live"
        )
        live_resp = QuestionGenerationResponse(
            resume_id=resume_id,
            request_id="req-diff4",
            trace_id="diff4-trace",
            question_id=live_q_id,
            question=live_output,
            validation_status="ACCEPTED",
            message="Live generated",
            attempts_count=1,
            latencies_ms={"total_ms": 110.0}
        )

        with patch("app.services.adaptive_selection_service.rag_pipeline.generate_grounded_question", return_value=live_resp):
            resp = engine.select_and_generate_next_question(session_id, db)

            assert resp.status == "GENERATED"
            assert resp.difficulty == 4
            assert resp.question_text == "Fresh difficulty 4 question?"
            # Mismatched difficulty prefetch was discarded!
            assert session.session_state["_prefetch_metrics"]["prefetches_discarded"] == 1
            assert "prefetched_question" not in session.session_state

    def test_prefetch_consumed_at_most_once_under_concurrency(self):
        """Simulate race: first call consumes prefetch; second concurrent call finds no prefetch and generates live."""
        resume_id = uuid4()
        session_id = uuid4()
        prefetch_q_id = uuid4()
        live_q_id = uuid4()

        # Shared session state
        shared_state = {
            "metadata": {"prefetch_enabled": True},
            "knowledge_state": {
                "topics": {"FastAPI": {"questions_asked_count": 1, "coverage_status": "QUESTION_ASKED"}},
                "projects": {},
                "relationships": [],
                "claims": [],
                "knowledge_gaps": [],
                "summary": {},
                "coverage": {}
            },
            "prefetched_question": {
                "question_id": str(prefetch_q_id),
                "question": {"question": "Prefetched Docker question", "category": "Implementation", "difficulty": 3},
                "selection": {"strategy": "UNCOVERED_TOPIC", "target_topic": "Docker"},
                "knowledge_state_version": "q:1_ans:1_eval:1_diff:3",
                "difficulty": 3,
                "strategy": "UNCOVERED_TOPIC",
                "created_at": datetime.utcnow().isoformat(),
                "consumed": False
            }
        }

        session = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            total_questions=1,
            session_state=shared_state
        )

        db = MagicMock()
        db.bind.dialect.name = "postgresql"
        db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = session
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.return_value = 1
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        engine = AdaptiveSelectionEngine()

        with patch.object(engine, "determine_next_question_target") as mock_target:
            mock_target.return_value = NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.UNCOVERED_TOPIC,
                target_topic="Docker",
                target_entity="Docker",
                category="Implementation",
                difficulty=3,
                rationale="Breadth"
            )

            # Request 1: Consumes the prefetched question!
            resp1 = engine.select_and_generate_next_question(session_id, db)
            assert resp1.question_id == prefetch_q_id
            assert resp1.latencies_ms.get("prefetch_hit") == 1.0

            # Request 2 (concurrent or immediately following):
            # Because session row lock protected consumption, prefetched_question is now gone!
            live_output = QuestionOutputSchema(
                question="Second request fresh live question?",
                category="Implementation",
                difficulty=3,
                grounding_score=0.9,
                relevance_score=0.9,
                technical_quality_score=0.9,
                trace_id="req2-trace",
                evidence_ids=["ev_1"],
                evidence_text="Live evidence",
                rationale="Live rationale"
            )
            live_resp = QuestionGenerationResponse(
                resume_id=resume_id,
                request_id="req2",
                trace_id="req2-trace",
                question_id=live_q_id,
                question=live_output,
                validation_status="ACCEPTED",
                message="Live generated",
                attempts_count=1,
                latencies_ms={"total_ms": 100.0}
            )

            with patch("app.services.adaptive_selection_service.rag_pipeline.generate_grounded_question", return_value=live_resp):
                resp2 = engine.select_and_generate_next_question(session_id, db)
                assert resp2.question_id == live_q_id
                assert resp2.latencies_ms.get("prefetch_hit") is None  # Not from prefetch!
                # Consumed count is strictly 1
                assert session.session_state["_prefetch_metrics"]["prefetches_consumed"] == 1
