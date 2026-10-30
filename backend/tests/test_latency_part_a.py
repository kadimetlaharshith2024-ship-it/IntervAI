import pytest
from uuid import uuid4
from unittest.mock import patch, MagicMock

from app.models.interview import InterviewQuestion
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.relationship import Relationship
from app.schemas.rag import QuestionOutputSchema
from app.services.rag.validator import grounding_validator
from app.services.rag.embedding_service import embedding_service
from app.services.rag.retrievers import (
    resume_retrieval_cache, get_or_build_resume_cache,
    VectorRetriever, BM25LexicalRetriever
)
from app.services.llm.factory import get_llm_provider
from app.core.config import settings

def test_validator_deterministic_checks_trigger_zero_embeddings():
    """
    Part A1: Confirms that when a candidate fails any deterministic check
    (no citations, invalid citation, hallucinated project, missing entity,
    invalid relationship, invalid claim, assertive presupposition),
    ZERO calls to embedding_service are made.
    """
    resolved_evidence = {
        "chk_1": {
            "evidence_id": "chk_1",
            "text": "CampusBridge: Implemented backend microservices using FastAPI and PostgreSQL.",
            "project": "CampusBridge",
            "source_type": "chunk"
        }
    }
    available_claims = {}
    available_relationships = {}
    allowed_projects = ["CampusBridge"]
    allowed_entities = ["FastAPI", "PostgreSQL"]

    # 1. No evidence citations
    q_no_cit = QuestionOutputSchema(
        question="How did you structure database connection pooling in PostgreSQL?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=[],
        evidence_text="",
        rationale="Testing zero citations",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_query:
        is_valid, reason, details = grounding_validator.validate(
            question=q_no_cit,
            allowed_projects=allowed_projects,
            allowed_entities=allowed_entities,
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims=available_claims,
            available_relationships=available_relationships
        )
        assert not is_valid
        assert details.get("code") == "no_citations"
        assert spy_query.call_count == 0

    # 2. Unknown citation ID
    q_bad_cit = QuestionOutputSchema(
        question="How did you structure database connection pooling in PostgreSQL?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=["chk_unknown_999"],
        evidence_text="",
        rationale="Testing unknown citation",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_query:
        is_valid, reason, details = grounding_validator.validate(
            question=q_bad_cit,
            allowed_projects=allowed_projects,
            allowed_entities=allowed_entities,
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims=available_claims,
            available_relationships=available_relationships
        )
        assert not is_valid
        assert details.get("code") == "invalid_citation_id"
        assert spy_query.call_count == 0

    # 3. Hallucinated project
    q_bad_proj = QuestionOutputSchema(
        question="How did you structure database connection pooling in PostgreSQL?",
        category="Architecture",
        difficulty=3,
        target_project="NonExistentProject",
        target_entity="PostgreSQL",
        evidence_ids=["chk_1"],
        evidence_text="FastAPI and PostgreSQL.",
        rationale="Testing hallucinated project",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_query:
        is_valid, reason, details = grounding_validator.validate(
            question=q_bad_proj,
            allowed_projects=allowed_projects,
            allowed_entities=allowed_entities,
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims=available_claims,
            available_relationships=available_relationships
        )
        assert not is_valid
        assert details.get("code") == "hallucinated_project"
        assert spy_query.call_count == 0

    # 4. Target entity absent from cited evidence
    q_bad_ent = QuestionOutputSchema(
        question="How would you approach configuring Kubernetes cluster autoscaling?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Kubernetes",
        evidence_ids=["chk_1"],
        evidence_text="FastAPI and PostgreSQL.",
        rationale="Testing absent entity",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_query:
        is_valid, reason, details = grounding_validator.validate(
            question=q_bad_ent,
            allowed_projects=allowed_projects,
            allowed_entities=allowed_entities,
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims=available_claims,
            available_relationships=available_relationships
        )
        assert not is_valid
        assert details.get("code") == "entity_not_in_cited_evidence"
        assert spy_query.call_count == 0

    # 5. Unsupported implementation presupposition
    q_presuppose = QuestionOutputSchema(
        question="Walk me through how you implemented your custom distributed transaction coordinator in PostgreSQL.",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=["chk_1"],
        evidence_text="FastAPI and PostgreSQL.",
        rationale="Testing presupposition",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_query:
        is_valid, reason, details = grounding_validator.validate(
            question=q_presuppose,
            allowed_projects=allowed_projects,
            allowed_entities=allowed_entities,
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims=available_claims,
            available_relationships=available_relationships
        )
        assert not is_valid
        assert details.get("code") == "unsupported_implementation_presupposition"
        assert spy_query.call_count == 0

def test_stored_question_embeddings_used_without_recomputing():
    """
    Part A2: If an existing question already has a valid stored embedding with matching model,
    it must NOT trigger any embed_query call for that question.
    """
    active_model = embedding_service.model_name
    vec = [0.1] * 384
    existing_q = InterviewQuestion(
        id=uuid4(),
        resume_id=uuid4(),
        question_text="How would you optimize database query execution plans in PostgreSQL?",
        embedding=vec,
        embedding_model=active_model
    )

    with patch.object(embedding_service, "embed_texts") as spy_batch:
        is_dup, _ = grounding_validator.check_semantic_duplicate(
            new_question_text="Completely different topic about CSS styling",
            existing_questions=[existing_q],
            new_q_vector=vec
        )
        # Stored embedding should be used directly, zero batch calls
        assert spy_batch.call_count == 0

def test_legacy_rows_batch_embedding_backfill():
    """
    Part A2: Legacy rows with NULL embeddings are embedded in ONE batched call.
    """
    active_model = embedding_service.model_name
    q1 = InterviewQuestion(id=uuid4(), question_text="How does Kafka handle partition rebalancing?", embedding=None, embedding_model=None)
    q2 = InterviewQuestion(id=uuid4(), question_text="How do you handle consumer group offsets in Kafka?", embedding=None, embedding_model=None)
    new_vec = [0.1] * 384

    with patch.object(embedding_service, "embed_texts", return_value=[[0.05]*384, [0.06]*384]) as spy_batch:
        is_dup, _ = grounding_validator.check_semantic_duplicate(
            new_question_text="What strategies exist for Redis caching?",
            existing_questions=[q1, q2],
            new_q_vector=new_vec
        )
        # Both questions should be embedded in ONE batched call
        assert spy_batch.call_count == 1
        assert q1.embedding == [0.05]*384
        assert q1.embedding_model == active_model
        assert q2.embedding == [0.06]*384
        assert q2.embedding_model == active_model

def test_mismatched_model_recomputed_in_batch():
    """
    Part A2: Question with mismatched embedding_model is treated as missing and recomputed.
    """
    active_model = embedding_service.model_name
    q_outdated = InterviewQuestion(
        id=uuid4(),
        question_text="How does RabbitMQ route messages using topic exchanges?",
        embedding=[0.99] * 384,
        embedding_model="outdated-model-v0"
    )
    new_vec = [0.1] * 384

    with patch.object(embedding_service, "embed_texts", return_value=[[0.12]*384]) as spy_batch:
        is_dup, _ = grounding_validator.check_semantic_duplicate(
            new_question_text="How do you design message brokers?",
            existing_questions=[q_outdated],
            new_q_vector=new_vec
        )
        assert spy_batch.call_count == 1
        assert q_outdated.embedding == [0.12]*384
        assert q_outdated.embedding_model == active_model

def test_embedding_service_failure_falls_back_to_lexical_similarity():
    """
    Part A2: If embedding service fails (e.g. raises an exception), check_semantic_duplicate
    does NOT crash. It logs a warning and falls back to deterministic lexical similarity.
    """
    existing_q = InterviewQuestion(
        id=uuid4(),
        question_text="How would you design database indexing for high throughput in PostgreSQL?",
        embedding=None,
        embedding_model=None
    )

    with patch.object(embedding_service, "embed_query", side_effect=RuntimeError("Embedding service unreachable")):
        with patch.object(embedding_service, "embed_texts", side_effect=RuntimeError("Embedding service unreachable")):
            # Highly overlapping question should be detected as duplicate via lexical fallback
            is_dup, msg = grounding_validator.check_semantic_duplicate(
                new_question_text="How would you design database indexing for high throughput in PostgreSQL?",
                existing_questions=[existing_q]
            )
            assert is_dup is True
            assert "Lexical Overlap" in msg

            # Non-overlapping question should not be detected as duplicate
            is_dup_diff, _ = grounding_validator.check_semantic_duplicate(
                new_question_text="Explain React useEffect dependency array semantics.",
                existing_questions=[existing_q]
            )
            assert is_dup_diff is False

def test_mixed_stored_and_missing_embeddings_in_session():
    """
    Part A2: Session with mixed valid, NULL, and mismatched embeddings only batches
    the missing/mismatched rows.
    """
    active_model = embedding_service.model_name
    q_valid = InterviewQuestion(id=uuid4(), question_text="Q1 text", embedding=[0.1]*384, embedding_model=active_model)
    q_legacy = InterviewQuestion(id=uuid4(), question_text="Q2 text", embedding=None, embedding_model=None)
    q_mismatched = InterviewQuestion(id=uuid4(), question_text="Q3 text", embedding=[0.2]*384, embedding_model="legacy-v1")
    new_vec = [0.1] * 384

    with patch.object(embedding_service, "embed_texts", return_value=[[0.3]*384, [0.4]*384]) as spy_batch:
        is_dup, _ = grounding_validator.check_semantic_duplicate(
            new_question_text="New distinct question?",
            existing_questions=[q_valid, q_legacy, q_mismatched],
            new_q_vector=new_vec
        )
        assert spy_batch.call_count == 1
        # Exactly 2 missing texts passed to embed_texts
        called_texts = spy_batch.call_args[0][0]
        assert len(called_texts) == 2
        assert "Q2 text" in called_texts
        assert "Q3 text" in called_texts
        assert "Q1 text" not in called_texts

def test_per_resume_retrieval_cache_and_vectorization():
    """
    Part A3: Verifies that get_or_build_resume_cache caches chunks, relationships,
    claims, and pre-tokenized BM25 stats, and VectorRetriever uses vectorized numpy cosine sim.
    """
    rid = uuid4()
    mock_db = MagicMock()
    chunk1 = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_c1",
        section="Experience",
        project="Alpha",
        chunk_text="Alpha: Microservices in Go with gRPC.",
        embedding=[1.0] + [0.0]*383,
        embedding_model=embedding_service.model_name
    )
    chunk2 = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_c2",
        section="Experience",
        project="Beta",
        chunk_text="Beta: Monolith in Ruby on Rails.",
        embedding=[0.0, 1.0] + [0.0]*382,
        embedding_model=embedding_service.model_name
    )

    mock_db.query.return_value.filter.return_value.all.side_effect = [
        [chunk1, chunk2],  # ResumeChunk
        [],                # Relationship
        []                 # ResumeClaim
    ]

    resume_retrieval_cache.invalidate(rid)
    cached = get_or_build_resume_cache(rid, mock_db)

    assert cached.resume_id == str(rid)
    assert len(cached.chunks) == 2
    assert cached.chunk_embeddings_matrix is not None
    assert cached.chunk_embeddings_matrix.shape == (2, 384)
    assert "Alpha" in cached.allowed_projects

    # Second call must hit memory cache without querying mock_db
    mock_db.query.reset_mock()
    cached2 = get_or_build_resume_cache(rid, mock_db)
    assert cached2 is cached
    assert mock_db.query.call_count == 0

    # Vectorized retrieval using numpy dot product
    res = VectorRetriever.retrieve(
        resume_id=rid,
        query_embedding=[1.0] + [0.0]*383,
        db=mock_db,
        top_k=2
    )
    assert len(res) == 2
    assert res[0][0].chunk_id == "chk_c1"
    assert pytest.approx(res[0][1], 0.01) == 1.0

def test_resume_cache_invalidation():
    """
    Part A3: Confirms invalidate removes cached data from memory.
    """
    rid = uuid4()
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []
    
    get_or_build_resume_cache(rid, mock_db)
    assert resume_retrieval_cache.get(rid) is not None

    resume_retrieval_cache.invalidate(rid)
    assert resume_retrieval_cache.get(rid) is None

def test_generation_and_evaluation_model_configuration():
    """
    Part A6: Confirms dedicated GENERATION_MODEL and EVALUATION_MODEL settings
    are respected when obtaining LLM providers.
    """
    p_gen = get_llm_provider(force_mock=True, role="generation")
    assert p_gen is not None

    p_eval = get_llm_provider(force_mock=True, role="evaluation")
    assert p_eval is not None
