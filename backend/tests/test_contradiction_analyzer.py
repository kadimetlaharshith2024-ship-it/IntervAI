from uuid import uuid4
import pytest

from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import InterviewQuestion
from app.schemas.contradiction import (
    ContradictionStatus,
    ContradictionType,
    ContradictionSeverity,
)
from app.services.evaluation.contradiction_analyzer import (
    DeterministicContradictionAnalyzer,
    contradiction_analyzer,
)


@pytest.fixture
def sample_evidence():
    """Provides structured sample relationships and claims for deterministic testing."""
    resume_id = uuid4()

    # Relationship 1: CampusBridge project uses PostgreSQL
    rel_db = Relationship(
        id=uuid4(),
        resume_id=resume_id,
        source_entity="CampusBridge",
        relation_type="BUILT_WITH",
        target_entity="PostgreSQL",
        section="Projects",
        context_entity="CampusBridge",
        evidence_text="Developed CampusBridge using PostgreSQL for relational entity persistence.",
        confidence=0.98
    )

    # Relationship 2: CampusBridge project containerized with Docker
    rel_docker = Relationship(
        id=uuid4(),
        resume_id=resume_id,
        source_entity="CampusBridge",
        relation_type="CONTAINERIZED_WITH",
        target_entity="Docker",
        section="Projects",
        context_entity="CampusBridge",
        evidence_text="Containerized microservices using Docker and orchestrated deployments.",
        confidence=0.96
    )

    # Relationship 3: CampusBridge caching with Redis
    rel_redis = Relationship(
        id=uuid4(),
        resume_id=resume_id,
        source_entity="CampusBridge",
        relation_type="BUILT_WITH",
        target_entity="Redis",
        section="Projects",
        context_entity="CampusBridge",
        evidence_text="Configured Redis cache-aside tier reducing database reads by 60%.",
        confidence=0.95
    )

    # Relationship 4: Work experience at TechCorp as Intern
    rel_role = Relationship(
        id=uuid4(),
        resume_id=resume_id,
        source_entity="Candidate",
        relation_type="WORKED_AS",
        target_entity="Software Engineer Intern",
        section="Experience",
        context_entity="TechCorp",
        evidence_text="Worked as Software Engineer Intern at TechCorp contributing to backend services.",
        confidence=0.97
    )

    # Relationship 5: Separate project TaskFlow using MongoDB
    rel_mongo = Relationship(
        id=uuid4(),
        resume_id=resume_id,
        source_entity="TaskFlow",
        relation_type="BUILT_WITH",
        target_entity="MongoDB",
        section="Projects",
        context_entity="TaskFlow",
        evidence_text="Built TaskFlow document management using MongoDB.",
        confidence=0.94
    )

    # Claim for CampusBridge PostgreSQL
    clm_db = ResumeClaim(
        id=uuid4(),
        resume_id=resume_id,
        claim_id="clm_pg_123",
        claim_text="Built CampusBridge using PostgreSQL.",
        section="Projects",
        project="CampusBridge",
        target_entity="PostgreSQL",
        relation_type="BUILT_WITH",
        evidence_text="Developed CampusBridge using PostgreSQL for relational entity persistence.",
        confidence=0.98,
        status="SUPPORTED"
    )

    # Question grounded in CampusBridge PostgreSQL
    q_cb = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_id,
        question_text="How did you configure indexing in PostgreSQL for CampusBridge to optimize search queries?",
        category="Implementation",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_text="Developed CampusBridge using PostgreSQL for relational entity persistence."
    )

    return {
        "resume_id": resume_id,
        "relationships": [rel_db, rel_docker, rel_redis, rel_role, rel_mongo],
        "claims": [clm_db],
        "q_cb": q_cb,
    }


# ==============================================================================
# Rule 1 Tests: Technology Substitution on Anchored Project
# ==============================================================================

def test_technology_substitution_direct_replacement_detected(sample_evidence):
    """Test 1: Explicit replacement of verified PostgreSQL with Oracle DBMS triggers substitution contradiction."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]
    clms = sample_evidence["claims"]

    answer = "In CampusBridge, we used Oracle DBMS instead of PostgreSQL for all relational storage."
    result = analyzer.analyze(answer, question=q, relationships=rels, claims=clms)

    assert result.has_contradiction is True
    assert result.status == ContradictionStatus.CONTRADICTION
    assert result.severity == ContradictionSeverity.HIGH
    assert len(result.conflicts) == 1

    conflict = result.conflicts[0]
    assert conflict.conflict_type == ContradictionType.TECHNOLOGY_SUBSTITUTION
    assert conflict.context_project == "CampusBridge"
    assert "Oracle DBMS" in conflict.candidate_assertion
    assert "PostgreSQL" in conflict.verified_fact


def test_technology_substitution_competing_primary_database_detected(sample_evidence):
    """Test 2: Explicitly asserting MySQL as primary database on CampusBridge triggers contradiction."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "For CampusBridge, our primary database was MySQL, where we configured InnoDB table locks."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is True
    assert result.status == ContradictionStatus.CONTRADICTION
    assert result.conflicts[0].conflict_type == ContradictionType.TECHNOLOGY_SUBSTITUTION
    assert "CampusBridge" in result.conflicts[0].context_project


def test_no_substitution_when_technologies_are_complementary(sample_evidence):
    """Test 3: Using Redis alongside PostgreSQL in CampusBridge is complementary, NOT contradictory."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "In CampusBridge, we used PostgreSQL for transactional persistence and Redis as a cache-aside layer."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.SUPPORTED
    assert "PostgreSQL" in result.supported_claims
    assert "Redis" in result.supported_claims


# ==============================================================================
# Rule 2 Tests: Explicit Negation of Verified Architecture
# ==============================================================================

def test_explicit_negation_of_containerization_detected(sample_evidence):
    """Test 4: Explicit denial that Docker was used triggers EXPLICIT_NEGATION contradiction."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "In our pipeline we did not use Docker or containerization; we deployed directly to bare-metal servers."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is True
    assert result.status == ContradictionStatus.CONTRADICTION
    assert result.severity == ContradictionSeverity.HIGH
    assert len(result.conflicts) == 1

    conflict = result.conflicts[0]
    assert conflict.conflict_type == ContradictionType.EXPLICIT_NEGATION
    assert "Docker" in conflict.verified_fact


def test_explicit_negation_passive_voice_detected(sample_evidence):
    """Test 5: Passive negation ('no Redis was used') triggers EXPLICIT_NEGATION contradiction."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "For this project, no Redis was configured because we relied entirely on in-memory process caches."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is True
    assert result.conflicts[0].conflict_type == ContradictionType.EXPLICIT_NEGATION


# ==============================================================================
# Rule 3 Tests: Role Incompatibility with Employer Context
# ==============================================================================

def test_role_incompatibility_intern_claiming_lead_architect(sample_evidence):
    """Test 6: Verified Software Engineer Intern claiming to be Lead Architect at TechCorp triggers ROLE_INCOMPATIBILITY."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "When I was the Lead Architect at TechCorp, I oversaw the entire infrastructure overhaul."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is True
    assert result.status == ContradictionStatus.CONTRADICTION
    assert result.severity == ContradictionSeverity.HIGH

    conflict = result.conflicts[0]
    assert conflict.conflict_type == ContradictionType.ROLE_INCOMPATIBILITY
    assert "TechCorp" in conflict.context_project
    assert "Software Engineer Intern" in conflict.verified_fact


def test_role_incompatibility_does_not_fire_without_clear_employer_match(sample_evidence):
    """Test 7: Claiming to be a Lead Architect without specifying TechCorp does not trigger a false conflict."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "As a lead architect in my consulting work, I evaluate database topologies carefully."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    # No TechCorp employer anchor -> no role contradiction
    assert result.has_contradiction is False
    assert not any(c.conflict_type == ContradictionType.ROLE_INCOMPATIBILITY for c in result.conflicts)


# ==============================================================================
# False-Positive Protection Tests (Step 4 Requirements)
# ==============================================================================

def test_unmentioned_auxiliary_tools_classified_as_unverified_additions(sample_evidence):
    """Test 8: Unmentioned auxiliary tools (Pytest, Nginx, Datadog) are UNVERIFIED_ADDITION, NOT contradictory."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "In CampusBridge, we wrote unit tests with Pytest and configured an Nginx reverse proxy for load balancing."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.UNVERIFIED_ADDITION
    assert len(result.conflicts) == 0


def test_conceptual_paraphrases_are_supported_not_contradictory(sample_evidence):
    """Test 9: Conceptual descriptions ('relational database', 'in-memory cache') align with verified technologies."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "In CampusBridge, we persisted accounts in our relational database and cached hot queries in an in-memory cache."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.SUPPORTED
    assert len(result.conflicts) == 0


def test_hypothetical_alternatives_and_tradeoffs_excluded_from_contradictions(sample_evidence):
    """Test 10: Trade-off and hypothetical sentences ('could have considered MySQL') do NOT trigger substitution."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = (
        "In CampusBridge we used PostgreSQL for data integrity. "
        "However, if we had required massive multi-region replication, we could have considered MySQL or CockroachDB as an alternative trade-off."
    )
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.SUPPORTED
    assert "PostgreSQL" in result.supported_claims
    assert len(result.hypothetical_claims) > 0
    assert len(result.conflicts) == 0


def test_purely_hypothetical_answer_classified_as_hypothetical_discussion(sample_evidence):
    """Test 11: An answer consisting entirely of hypothetical trade-offs is HYPOTHETICAL_DISCUSSION."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    answer = "If we were evaluating database choices today, an alternative approach would be using MongoDB or CockroachDB to weigh read vs write trade-offs."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.HYPOTHETICAL_DISCUSSION
    assert len(result.conflicts) == 0


def test_general_theoretical_fundamentals_question_exempt(sample_evidence):
    """Test 12: General theoretical question (category=Fundamentals) does not trigger project contradiction rules."""
    analyzer = DeterministicContradictionAnalyzer()
    rels = sample_evidence["relationships"]

    q_theory = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_evidence["resume_id"],
        question_text="Compare optimistic locking with pessimistic locking in distributed databases.",
        category="Fundamentals",
        difficulty=3,
        target_project=None,
        target_entity=None,
        evidence_text=""
    )

    answer = "Optimistic locking checks version timestamps, whereas pessimistic locking uses SELECT FOR UPDATE in Oracle or PostgreSQL."
    result = analyzer.analyze(answer, question=q_theory, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.NO_APPLICABLE_EVIDENCE
    assert len(result.conflicts) == 0


def test_technology_mentioned_in_different_project_does_not_conflict(sample_evidence):
    """Test 13: Mentioning MongoDB for TaskFlow does not conflict with CampusBridge's PostgreSQL."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]  # Target project is CampusBridge
    rels = sample_evidence["relationships"]

    answer = "In CampusBridge we used PostgreSQL, while in our other project TaskFlow we used MongoDB for unstructured records."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.SUPPORTED
    assert "PostgreSQL" in result.supported_claims
    assert "MongoDB" in result.supported_claims
    assert len(result.conflicts) == 0


def test_missing_or_ambiguous_project_context_produces_no_contradiction(sample_evidence):
    """Test 14: Mentioning MySQL without anchoring to CampusBridge is UNVERIFIED_ADDITION, not a contradiction."""
    analyzer = DeterministicContradictionAnalyzer()
    q = sample_evidence["q_cb"]
    rels = sample_evidence["relationships"]

    # Candidate makes a generic statement without anchoring to CampusBridge
    answer = "I have previously configured MySQL master-slave replication and tuned query buffers."
    result = analyzer.analyze(answer, question=q, relationships=rels)

    assert result.has_contradiction is False
    assert result.status == ContradictionStatus.UNVERIFIED_ADDITION
    assert len(result.conflicts) == 0
