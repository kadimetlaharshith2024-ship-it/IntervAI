import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from app.services.knowledge_state_service import calculate_difficulty_aware_mastery, knowledge_state_service
from app.services.rag.validator import IndependentGroundingValidator
from app.services.adaptive_selection_service import (
    AdaptiveSelectionEngine,
    AdaptiveSelectionConfig,
    AdaptiveStrategyType,
)
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.models.rag import ResumeClaim
from app.schemas.rag import QuestionOutputSchema


class TestPartBMasteryBoundedProperties:
    """Tests for Part B2: Bounded difficulty-aware mastery properties."""

    def test_property_1_60_diff5_not_higher_than_80_diff1(self):
        """(1) A 60 at difficulty 5 must NOT yield higher mastery than an 80 at difficulty 1."""
        mastery_60_d5 = calculate_difficulty_aware_mastery([], 60.0, 5)
        mastery_80_d1 = calculate_difficulty_aware_mastery([], 80.0, 1)

        assert mastery_60_d5 < mastery_80_d1
        # Specifically: 62.0 < 76.0
        assert mastery_60_d5 == 62.0
        assert mastery_80_d1 == 76.0

    def test_property_2_identical_scores_differ_by_no_more_than_documented_bound(self):
        """(2) Identical scores at different difficulties differ by no more than the documented bound (+/- 5 points)."""
        for score in [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]:
            diff_results = [calculate_difficulty_aware_mastery([], score, d) for d in range(1, 6)]
            min_val = min(diff_results)
            max_val = max(diff_results)
            assert max_val - min_val <= 5.01, f"Score {score} had span {max_val - min_val} > 5.0"

    def test_property_3_no_input_combination_requires_clamping(self):
        """(3) No input combination requires clamping: all outputs naturally in [0, 100]."""
        for d in range(1, 6):
            for score in [0.0, 0.1, 50.0, 99.9, 100.0]:
                val = calculate_difficulty_aware_mastery([], score, d)
                assert 0.0 <= val <= 100.0
                # Multiple history combinations
                val_hist = calculate_difficulty_aware_mastery(
                    [{"score": 0.0, "difficulty": 1}, {"score": 100.0, "difficulty": 5}],
                    score,
                    d
                )
                assert 0.0 <= val_hist <= 100.0

    def test_property_4_mastery_is_monotonic_in_score_at_fixed_difficulty(self):
        """(4) Mastery is strictly monotonic in score at fixed difficulty."""
        for d in range(1, 6):
            prev_mastery = -1.0
            for score in range(0, 105, 5):
                mastery = calculate_difficulty_aware_mastery([], float(score), d)
                assert mastery > prev_mastery, f"Monotonicity failed at d={d}, score={score}"
                prev_mastery = mastery


class TestPartBDifficultyValidation:
    """Tests for Part B4: Multi-marker difficulty estimation and all-level 2-step validation."""

    def test_estimate_difficulty_level_examples_1_to_5(self):
        """Estimate difficulty level accurately weighs markers for Levels 1 through 5."""
        val = IndependentGroundingValidator()

        # Level 1: Basic definitions/syntax
        l1_q = "What is an index in PostgreSQL and what is the definition of a primary key?"
        assert val.estimate_difficulty_level(l1_q) == 1

        # Level 2: Straightforward integration/explanation
        l2_q = "Explain how to integrate Redis into a Spring Boot service and describe how data flows."
        assert val.estimate_difficulty_level(l2_q) == 2

        # Level 3: Concrete design/lifecycle/transactions
        l3_q = "How would you design cache invalidation and manage transaction boundaries in PostgreSQL?"
        assert val.estimate_difficulty_level(l3_q) == 3

        # Level 4: Architecture trade-offs/decoupling/scaling
        l4_q = "What architectural trade-offs would you evaluate when decoupling services with asynchronous data flows to solve horizontal scale bottlenecks?"
        assert val.estimate_difficulty_level(l4_q) == 4

        # Level 5: Distributed failure modes/partitions/extreme load
        l5_q = "How do you mitigate cascading failure modes and preserve distributed consistency during a network partition with split-brain risks under extreme load?"
        assert val.estimate_difficulty_level(l5_q) == 5

    def test_difficulty_validation_rejects_difference_ge_2_in_either_direction(self):
        """Reject if requested and estimated difficulty differ by 2 or more levels in EITHER direction."""
        validator = IndependentGroundingValidator()

        # Case 1: Requested 5, estimated 2 or 3 (>= 2 level drop)
        q_under = QuestionOutputSchema(
            question="Explain how to integrate Redis and describe how data flows.",  # Level 2
            category="Architecture",
            difficulty=5,  # Requested 5
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="test-trace",
            evidence_ids=["chk_1"],
            evidence_text="Uses Redis for cache.",
            rationale="Test"
        )
        is_valid, reason, details = validator.validate(
            question=q_under,
            allowed_projects=["TestProj"],
            allowed_entities=["Redis"],
            resolved_evidence_items={"chk_1": {"evidence_id": "chk_1", "text": "Uses Redis for cache."}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        assert is_valid is False
        assert details.get("code") == "difficulty_mismatch"

        # Case 2: Requested 1, estimated 4 (>= 2 level jump)
        q_over = QuestionOutputSchema(
            question="What architectural trade-offs would you evaluate when decoupling services with asynchronous data flows to solve horizontal scale bottlenecks?",  # Level 4
            category="Architecture",
            difficulty=1,  # Requested 1
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="test-trace",
            evidence_ids=["chk_1"],
            evidence_text="Decoupled services using async event flows to solve horizontal scale bottlenecks.",
            rationale="Test"
        )
        is_valid, reason, details = validator.validate(
            question=q_over,
            allowed_projects=["TestProj"],
            allowed_entities=["Redis"],
            resolved_evidence_items={"chk_1": {"evidence_id": "chk_1", "text": "Decoupled services using async event flows."}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        assert is_valid is False
        assert details.get("code") == "difficulty_mismatch"


class TestPartBGapResolutionAndRanking:
    """Tests for Part B5: Gap resolution precision and severity ranking."""

    def test_gap_resolution_requires_same_question_or_covered_missing_concepts(self):
        """A score >= 75 should only close gaps tied to the same question_id or same topic with covered concepts."""
        engine = knowledge_state_service

        resume_id = uuid4()
        session = MagicMock(spec=InterviewSession)
        session.id = uuid4()
        session.resume_id = resume_id
        session.session_state = {
            "knowledge_state": {
                "topics": {
                    "PostgreSQL": {
                        "topic_id": "PostgreSQL",
                        "topic_name": "PostgreSQL",
                        "historical_scores": [40.0],
                        "mastery_score": 40.0,
                        "confidence": 0.5,
                        "coverage_status": "REQUIRES_FOLLOW_UP",
                        "assessment_status": "ASSESSED",
                        "answers_evaluated_count": 1,
                        "questions_asked_count": 1
                    }
                },
                "projects": {},
                "relationships": [],
                "claims": [],
                "knowledge_gaps": [
                    {
                        "gap_id": "gap_pg_1",
                        "topic": "PostgreSQL",
                        "question_id": "q_original_123",
                        "description": "missing WAL mechanics; missing vacuum tuning",
                        "missing_concepts": ["WAL mechanics", "vacuum tuning"],
                        "severity": "HIGH",
                        "is_addressed": False,
                        "created_at": "2026-01-01T00:00:00"
                    }
                ],
                "summary": {},
                "coverage": {},
                "_processed_answer_ids": [],
                "_question_latest_attempts": {}
            }
        }

        # Scenario 1: New question on PostgreSQL has score 80, but candidate STILL misses "WAL mechanics"
        q_new = MagicMock(spec=InterviewQuestion)
        q_new.id = uuid4()
        q_new.resume_id = resume_id
        q_new.target_entity = "PostgreSQL"
        q_new.target_project = None
        q_new.category = "Architecture"
        q_new.difficulty = 3
        q_new.expected_concepts = ["WAL mechanics"]

        db = MagicMock()
        def mock_db_query(model):
            m = MagicMock()
            if model == InterviewSession:
                m.filter.return_value.first.return_value = session
            elif model == InterviewQuestion:
                m.filter.return_value.first.return_value = q_new
            else:
                m.filter.return_value.all.return_value = []
                m.filter.return_value.first.return_value = None
            return m
        db.query.side_effect = mock_db_query

        ans_still_missing = MagicMock(spec=InterviewAnswer)
        ans_still_missing.id = uuid4()
        ans_still_missing.overall_score = 80.0
        ans_still_missing.evaluation_confidence = 0.9
        ans_still_missing.missing_concepts = ["WAL mechanics"]  # Still missing!
        ans_still_missing.weaknesses = []
        ans_still_missing.attempt_number = 1
        ans_still_missing.status = "EVALUATED"

        updated_state = engine.update_from_evaluation(session.id, q_new.id, ans_still_missing, db)
        gaps = updated_state["knowledge_gaps"]
        old_gap = next(g for g in gaps if g["gap_id"] == "gap_pg_1")
        # Old gap must NOT be marked addressed because missing concept WAL mechanics was NOT covered!
        assert old_gap["is_addressed"] is False

        # Scenario 2: Another question on PostgreSQL scores 85 and candidate covers both concepts (no missing concepts)
        ans_covered = MagicMock(spec=InterviewAnswer)
        ans_covered.id = uuid4()
        ans_covered.overall_score = 85.0
        ans_covered.evaluation_confidence = 0.9
        ans_covered.missing_concepts = []  # None missing!
        ans_covered.matched_concepts = ["WAL mechanics", "vacuum tuning"]
        ans_covered.weaknesses = []
        ans_covered.attempt_number = 1
        ans_covered.status = "EVALUATED"

        updated_state_2 = engine.update_from_evaluation(session.id, q_new.id, ans_covered, db)
        old_gap_2 = next(g for g in updated_state_2["knowledge_gaps"] if g["gap_id"] == "gap_pg_1")
        # Now it is properly resolved!
        assert old_gap_2["is_addressed"] is True


class TestPartBRollingWindowDifficultyAdaptation:
    """Tests for Part B2: Rolling window difficulty escalation and de-escalation."""

    def test_escalation_requires_latest_and_window_avg_above_threshold(self):
        """Escalation occurs only if latest score > high_th AND rolling window avg > high_th."""
        engine = AdaptiveSelectionEngine()

        session = MagicMock(spec=InterviewSession)
        session.id = uuid4()
        session.status = "IN_PROGRESS"
        session.current_difficulty = 3
        session.session_state = {
            "knowledge_state": {
                "topics": {},
                "projects": {},
                "relationships": [],
                "claims": [],
                "knowledge_gaps": [],
                "summary": {},
                "coverage": {}
            }
        }

        q1 = MagicMock(spec=InterviewQuestion, id=uuid4(), parent_question_id=None, target_entity="Python", category="Implementation", target_project=None)
        q2 = MagicMock(spec=InterviewQuestion, id=uuid4(), parent_question_id=None, target_entity="Python", category="Implementation", target_project=None)
        q3 = MagicMock(spec=InterviewQuestion, id=uuid4(), parent_question_id=None, target_entity="Python", category="Implementation", target_project=None)

        # Scenario A: Scores are [50, 60, 90]. Latest is 90 (> 75), but window avg is (50+60+90)/3 = 66.7 (<= 75).
        # Difficulty must NOT escalate!
        a1 = MagicMock(spec=InterviewAnswer, status="EVALUATED", overall_score=50.0, evaluation_confidence=0.9)
        a2 = MagicMock(spec=InterviewAnswer, status="EVALUATED", overall_score=60.0, evaluation_confidence=0.9)
        a3 = MagicMock(spec=InterviewAnswer, status="EVALUATED", overall_score=90.0, evaluation_confidence=0.9)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.return_value = 3
        db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
            [q1, q2, q3],  # questions
            [a1, a2, a3],  # answers
        ]

        cfg = AdaptiveSelectionConfig(
            high_performance_threshold=75.0,
            low_performance_threshold=50.0,
            rolling_window_size=3,
            min_confidence_for_escalation=0.65
        )

        res = engine.determine_next_question_target(session.id, db, config_override=cfg)
        assert res.difficulty == 3  # Maintained because window avg 66.7 <= 75.0

        # Scenario B: Scores are [80, 85, 90]. Window avg is 85.0 (> 75).
        # Difficulty MUST escalate to 4!
        a1.overall_score = 80.0
        a2.overall_score = 85.0
        db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
            [q1, q2, q3],
            [a1, a2, a3],
        ]

        res_esc = engine.determine_next_question_target(session.id, db, config_override=cfg)
        assert res_esc.difficulty == 4


class TestPartBBreadthQuota:
    """Tests for Part B1: Breadth quota reserving share for claims and project depth."""

    def test_quota_prefers_unmet_claim_when_budget_is_tight(self):
        """When question budget is tight and claim quota is unmet, claim verification is preferred over uncovered topic."""
        engine = AdaptiveSelectionEngine()

        session = MagicMock(spec=InterviewSession)
        session.id = uuid4()
        session.status = "IN_PROGRESS"
        session.current_difficulty = 3
        session.session_state = {
            "knowledge_state": {
                "topics": {
                    "FastAPI": {"questions_asked_count": 0, "answers_evaluated_count": 0, "coverage_status": "NOT_CONSIDERED"},
                    "Docker": {"questions_asked_count": 0, "answers_evaluated_count": 0, "coverage_status": "NOT_CONSIDERED"}
                },
                "projects": {
                    "IntervAI": {"remaining_areas_to_assess": ["Architecture"], "questions_asked_count": 0}
                },
                "relationships": [],
                "claims": [
                    {"claim_id": "clm_1", "project": "IntervAI", "target_entity": "FastAPI", "claim_text": "Built real-time streaming pipeline", "status": "UNTESTED"}
                ],
                "knowledge_gaps": [],
                "summary": {},
                "coverage": {}
            }
        }

        # Suppose 8 questions have been asked out of a budget of 10.
        # Remaining budget = 2.
        # Claim quota = 20% of 10 = 2 questions. Claims asked = 0. Deficit = 2.
        # Budget is tight (remaining 2 <= deficit 2).
        # Must pick CLAIM_VERIFICATION instead of UNCOVERED_TOPIC!
        questions = [MagicMock(spec=InterviewQuestion, target_claim_id=None, category="Architecture", targeted_claim="", target_project=None, target_entity="Topic") for _ in range(8)]
        for q in questions:
            q.parent_question_id = None

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = session
        db.query.return_value.filter.return_value.count.return_value = 8
        db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
            questions,  # 8 questions
            [],         # answers
        ]

        cfg = AdaptiveSelectionConfig(
            default_max_session_questions=10,
            claim_verification_quota_percent=0.20,
            project_depth_quota_percent=0.20
        )

        res = engine.determine_next_question_target(session.id, db, config_override=cfg)
        assert res.strategy == AdaptiveStrategyType.CLAIM_VERIFICATION
        assert res.target_claim_id == "clm_1"


class TestPartBSessionScopedDuplicates:
    """Tests for Part B3: Session-scoped duplicate rejection."""

    def test_past_session_questions_not_hard_rejected(self):
        """Questions from previous sessions are never hard-rejected; only current-session questions are."""
        resume_id = uuid4()
        session_id_1 = uuid4()  # Previous session
        session_id_2 = uuid4()  # Current session

        # Previous session asked about Redis caching
        past_q = InterviewQuestion(
            id=uuid4(),
            resume_id=resume_id,
            session_id=session_id_1,
            question_text="How would you design cache invalidation in Redis for high throughput?",
            difficulty=3,
            category="Architecture"
        )
        current_q = InterviewQuestion(
            id=uuid4(),
            resume_id=resume_id,
            session_id=session_id_2,
            question_text="Describe Postgres index types.",
            difficulty=2,
            category="Implementation"
        )

        all_resume_questions = [past_q, current_q]

        # Verify pipeline session scoping logic
        existing_questions = [q for q in all_resume_questions if q.session_id == session_id_2]
        past_session_questions = [q for q in all_resume_questions if q.session_id != session_id_2]

        assert len(existing_questions) == 1
        assert existing_questions[0].id == current_q.id
        assert len(past_session_questions) == 1
        assert past_session_questions[0].id == past_q.id

        # Verify validator duplicate rejection behavior:
        cand = QuestionOutputSchema(
            question="How would you design cache invalidation in Redis for high throughput?",
            category="Architecture",
            difficulty=3,
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="trace-1",
            evidence_ids=["chk_1"],
            evidence_text="Redis cache",
            rationale="Test"
        )
        val = IndependentGroundingValidator()

        # 1. Against current session (which only contains current_q) -> VALID (not rejected as duplicate!)
        is_valid, reason, _ = val.validate(
            question=cand,
            allowed_projects=["TestProj"],
            allowed_entities=["Redis"],
            resolved_evidence_items={"chk_1": {"evidence_id": "chk_1", "text": "Redis cache"}},
            existing_questions=existing_questions,
            available_claims={},
            available_relationships={}
        )
        assert is_valid is True

        # 2. If it were checked against the entire history across sessions (existing_questions = all_resume_questions) -> REJECTED!
        is_valid_dup, reason_dup, details_dup = val.validate(
            question=cand,
            allowed_projects=["TestProj"],
            allowed_entities=["Redis"],
            resolved_evidence_items={"chk_1": {"evidence_id": "chk_1", "text": "Redis cache"}},
            existing_questions=all_resume_questions,
            available_claims={},
            available_relationships={}
        )
        assert is_valid_dup is False
        assert details_dup.get("code") == "semantic_duplicate"

