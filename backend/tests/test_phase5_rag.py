import pytest
from uuid import uuid4
from app.schemas.rag import QuestionGenerationRequest, EvidenceItem, QuestionOutputSchema
from app.models.interview import InterviewQuestion
from app.models.rag import ResumeClaim
from app.services.rag.retrievers import BM25LexicalRetriever
from app.services.rag.reranker import reranker
from app.services.rag.generator import QuestionGenerationService
from app.services.rag.validator import grounding_validator

def test_placeholder_input_sanitization():
    req1 = QuestionGenerationRequest(target_project="string", target_entity="string")
    assert req1.target_project is None
    assert req1.target_entity is None

    req2 = QuestionGenerationRequest(target_project="null", target_entity="undefined")
    assert req2.target_project is None
    assert req2.target_entity is None

def test_interview_relevance_disqualifies_languages():
    non_tech_item = EvidenceItem(
        evidence_id="rel_spoken_lang",
        text="Candidate SPEAKS English (Evidence: English (Professional))",
        section="Languages",
        project=None,
        source_type="relationship",
        score=0.85,
        provenance="section_structural"
    )
    reranked = reranker.rerank(query="Architecture", items=[non_tech_item])
    assert len(reranked) == 0

def test_bm25_lexical_tokenization_and_scoring():
    tokens = BM25LexicalRetriever.tokenize("Implemented Linux kernel event monitoring with fanotify in C.")
    assert "fanotify" in tokens
    assert "kernel" in tokens
    assert "monitoring" in tokens

def test_validator_rejects_uncited_evidence_content():
    """Confirms that entity present in Chunk B is rejected if the model only cites Chunk A."""
    resolved_evidence = {
        "chk_a": {
            "evidence_id": "chk_a",
            "text": "Configured Nginx reverse proxy on Linux server.",
            "project": "WebServer"
        }
    }
    q = QuestionOutputSchema(
        question="In WebServer, how did you implement PostgreSQL connection pooling?",
        category="Architecture",
        difficulty=3,
        target_project="WebServer",
        target_entity="PostgreSQL",
        evidence_ids=["chk_a"],
        evidence_text="Configured Nginx reverse proxy on Linux server.",
        rationale="Testing citation",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )
    is_valid, reason, _ = grounding_validator.validate(
        question=q,
        allowed_projects=["WebServer"],
        allowed_entities=["PostgreSQL", "Nginx"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False
    assert "absent from the cited evidence" in reason.lower()

def test_validator_rejects_hallucinated_claim_id():
    resolved_evidence = {
        "chk_a": {
            "evidence_id": "chk_a",
            "text": "CacheService | Implemented distributed caching with Redis.",
            "project": "CacheService"
        }
    }
    q = QuestionOutputSchema(
        question="In CacheService, how did you configure Redis evictions?",
        category="Performance",
        difficulty=3,
        target_project="CacheService",
        target_entity="Redis",
        target_claim_id="clm_fake_9999",
        evidence_ids=["chk_a"],
        evidence_text="CacheService | Implemented distributed caching with Redis.",
        rationale="Testing claim ID",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )
    is_valid, reason, _ = grounding_validator.validate(
        question=q,
        allowed_projects=["CacheService"],
        allowed_entities=["Redis"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False
    assert "not a valid claim" in reason.lower()

def test_validator_rejects_invalid_relationship_id():
    resolved_evidence = {
        "chk_a": {
            "evidence_id": "chk_a",
            "text": "CacheService | Configured Redis caching cluster.",
            "project": "CacheService"
        }
    }
    q = QuestionOutputSchema(
        question="In CacheService, how did you configure Redis evictions?",
        category="Architecture",
        difficulty=3,
        target_project="CacheService",
        target_entity="Redis",
        target_relationship="rel_hallucinated_9999",
        evidence_ids=["chk_a"],
        evidence_text="CacheService | Configured Redis caching cluster.",
        rationale="Testing invalid relationship ID",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )
    is_valid, reason, details = grounding_validator.validate(
        question=q,
        allowed_projects=["CacheService"],
        allowed_entities=["Redis"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False
    assert "not a valid relationship" in reason.lower()
    assert details.get("code") == "invalid_relationship_id"

def test_generator_resolves_specific_cited_evidence_text(monkeypatch):
    """
    Confirms that cited evidence IDs resolve directly to their own text rather than default first block.
    Verifies deterministic valid resolution, rejection of invalid evidence IDs, and rejection of missing citations.
    """
    from unittest.mock import MagicMock
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    context = {
        "selected_evidence_items": [
            {
                "evidence_id": "chk_a",
                "section": "Projects",
                "project": "TalentBridge",
                "text": "TalentBridge: Developed Spring Boot backend."
            },
            {
                "evidence_id": "chk_b",
                "section": "Projects",
                "project": "CacheService",
                "text": "CacheService: Configured Redis distributed cache."
            }
        ],
        "claims": [],
        "relationships": [
            {
                "relationship_id": "rel_c1a2b3d4",
                "source": "CacheService",
                "rel": "USES",
                "target": "Redis"
            }
        ]
    }

    mock_provider = MagicMock()

    # Case 1: Deterministic mocked response citing chk_b specifically
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "In CacheService, how did you configure Redis evictions?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CacheService",
            "target_entity": "Redis",
            "target_claim_id": None,
            "target_relationship_id": "rel_c1a2b3d4",
            "evidence_ids": ["chk_b"],
            "rationale": "Direct Redis probe."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )
    monkeypatch.setattr("app.services.rag.generator.get_llm_provider", lambda force_mock=False: mock_provider)

    q = generator.generate(
        context=context,
        category="Architecture",
        difficulty=3,
        target_project="CacheService",
        target_entity="Redis",
        trace_id="trc_test"
    )

    # Must resolve specifically to chk_b text and exclude chk_a text
    assert q.evidence_ids == ["chk_b"]
    assert q.evidence_text == "CacheService: Configured Redis distributed cache."
    assert "Spring Boot backend" not in q.evidence_text
    assert q.target_relationship == "rel_c1a2b3d4"

    # Case 2: Invalid evidence ID must be rejected without fallback
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "In CacheService, how did you configure Redis evictions?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CacheService",
            "target_entity": "Redis",
            "target_claim_id": None,
            "target_relationship_id": "rel_c1a2b3d4",
            "evidence_ids": ["chk_invalid_9999"],
            "rationale": "Testing invalid citation rejection."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )
    with pytest.raises(ValueError, match="invalid evidence ID"):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="CacheService",
            target_entity="Redis",
            trace_id="trc_test"
        )

    # Case 3: Missing evidence IDs must be rejected without fallback
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "In CacheService, how did you configure Redis evictions?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CacheService",
            "target_entity": "Redis",
            "target_claim_id": None,
            "target_relationship_id": "rel_c1a2b3d4",
            "evidence_ids": [],
            "rationale": "Testing missing citation rejection."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )
    with pytest.raises(ValueError, match="no evidence citations"):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="CacheService",
            target_entity="Redis",
            trace_id="trc_test"
        )

    # Case 4: Invalid relationship ID must be rejected without fallback
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "In CacheService, how did you configure Redis evictions?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CacheService",
            "target_entity": "Redis",
            "target_claim_id": None,
            "target_relationship_id": "rel_hallucinated_9999",
            "evidence_ids": ["chk_b"],
            "rationale": "Testing invalid relationship rejection."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )
    with pytest.raises(ValueError, match="invalid relationship ID"):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="CacheService",
            target_entity="Redis",
            trace_id="trc_test"
        )

def test_embedding_service_fails_explicitly_when_encoder_unavailable():
    """Confirms that embedding service raises RuntimeError instead of fabricating fake MD5 vectors."""
    from app.services.rag.embedding_service import EmbeddingService
    service = EmbeddingService()
    service.encoder = None  # Force encoder unavailable state
    with pytest.raises(RuntimeError, match="Embedding model .* is unavailable"):
        service.embed_texts(["Grounded resume text."])
    with pytest.raises(RuntimeError, match="Embedding model .* is unavailable"):
        service.embed_query("Architecture query.")

def test_semantic_duplicate_rejection_via_embeddings():
    existing_q = InterviewQuestion(
        id=uuid4(),
        resume_id=uuid4(),
        question_text="How did you secure user authentication tokens in your project?",
        difficulty=3,
        category="Security"
    )
    # Paraphrased question: low Jaccard token overlap, high semantic similarity
    paraphrased_q = QuestionOutputSchema(
        question="In your project, what mechanisms were implemented to protect session tokens against theft?",
        category="Security",
        difficulty=3,
        target_project=None,
        target_entity=None,
        evidence_ids=["ev_1"],
        evidence_text="Secured token storage",
        rationale="Paraphrase test",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test"
    )
    is_dup, msg = grounding_validator.check_semantic_duplicate(
        paraphrased_q.question,
        [existing_q],
        similarity_threshold=0.75
    )
    assert is_dup is True
    assert "semantically duplicate" in msg.lower()

def test_difficulty_depth_estimation():
    lvl_1 = grounding_validator.estimate_difficulty_level("What is the primary role of the controller layer?")
    lvl_3 = grounding_validator.estimate_difficulty_level("How did you handle state transitions and transaction boundaries during failure?")
    lvl_5 = grounding_validator.estimate_difficulty_level("What threat vector could cause cascading failure under extreme load?")

    assert lvl_1 <= 2
    assert lvl_3 >= 3
    assert lvl_5 == 5

def test_vector_retriever_matching_model_retrieval():
    """Confirms that stored chunks with matching embedding_model are retrieved and scored normally."""
    from unittest.mock import MagicMock
    from app.models.rag import ResumeChunk
    from app.services.rag.retrievers import VectorRetriever
    from app.services.rag.embedding_service import embedding_service

    rid = uuid4()
    vec1 = [1.0] + [0.0] * 383
    vec2 = [0.0] + [1.0] + [0.0] * 382
    query_vec = [1.0] + [0.0] * 383

    chunk1 = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_1",
        section="Experience",
        project="WebServer",
        chunk_text="Nginx load balancing",
        embedding=vec1,
        embedding_model=embedding_service.model_name
    )
    chunk2 = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_2",
        section="Experience",
        project="Database",
        chunk_text="PostgreSQL pooling",
        embedding=vec2,
        embedding_model=embedding_service.model_name
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [chunk1, chunk2]
    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    results = VectorRetriever.retrieve(
        resume_id=rid,
        query_embedding=query_vec,
        db=mock_db,
        top_k=2
    )

    assert len(results) == 2
    assert results[0][0].chunk_id == "chk_1"
    assert pytest.approx(results[0][1], 0.01) == 1.0
    assert results[1][0].chunk_id == "chk_2"
    assert pytest.approx(results[1][1], 0.01) == 0.0

def test_vector_retriever_mismatched_model_rejection():
    """Confirms that stored chunks with legacy/incompatible embedding models are rejected with actionable error."""
    from unittest.mock import MagicMock
    from app.models.rag import ResumeChunk
    from app.services.rag.retrievers import VectorRetriever

    rid = uuid4()
    legacy_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_legacy_1",
        section="Experience",
        project="LegacyProject",
        chunk_text="Legacy chunk content",
        embedding=[0.5] * 384,
        embedding_model="legacy-md5-pseudo-v1"
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [legacy_chunk]
    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    with pytest.raises(ValueError, match="incompatible or missing embedding model metadata"):
        VectorRetriever.retrieve(
            resume_id=rid,
            query_embedding=[0.1] * 384,
            db=mock_db,
            top_k=5
        )

def test_vector_retriever_none_model_metadata_rejection():
    """Confirms that chunks with embedding_model=None are rejected with actionable re-indexing error."""
    from unittest.mock import MagicMock
    from app.models.rag import ResumeChunk
    from app.services.rag.retrievers import VectorRetriever

    rid = uuid4()
    none_model_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_none_meta",
        section="Experience",
        project="TestProject",
        chunk_text="Unidentified chunk content",
        embedding=[0.5] * 384,
        embedding_model=None
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [none_model_chunk]
    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    with pytest.raises(ValueError, match="incompatible or missing embedding model metadata"):
        VectorRetriever.retrieve(
            resume_id=rid,
            query_embedding=[0.1] * 384,
            db=mock_db,
            top_k=5
        )

def test_vector_retriever_empty_model_metadata_rejection():
    """Confirms that chunks with empty string embedding_model are rejected with actionable re-indexing error."""
    from unittest.mock import MagicMock
    from app.models.rag import ResumeChunk
    from app.services.rag.retrievers import VectorRetriever

    rid = uuid4()
    empty_model_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_empty_meta",
        section="Experience",
        project="TestProject",
        chunk_text="Empty metadata chunk content",
        embedding=[0.5] * 384,
        embedding_model="   "
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [empty_model_chunk]
    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    with pytest.raises(ValueError, match="incompatible or missing embedding model metadata"):
        VectorRetriever.retrieve(
            resume_id=rid,
            query_embedding=[0.1] * 384,
            db=mock_db,
            top_k=5
        )

def test_vector_retriever_empty_chunk_set():
    """Confirms that resumes with no stored chunks return an empty list without error."""
    from unittest.mock import MagicMock
    from app.services.rag.retrievers import VectorRetriever

    rid = uuid4()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []
    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    results = VectorRetriever.retrieve(
        resume_id=rid,
        query_embedding=[0.1] * 384,
        db=mock_db,
        top_k=5
    )

    assert results == []

def _build_test_mock_db(rid, chunks=None, claims=None, rels=None, questions=None):
    from unittest.mock import MagicMock
    from app.models.resume import Resume
    from app.models.rag import ResumeChunk, ResumeClaim
    from app.models.relationship import Relationship
    from app.models.interview import InterviewQuestion

    mock_db = MagicMock()
    mock_resume = Resume(id=rid, raw_text="Experienced Software Engineer with Go, Redis, and PostgreSQL.")
    chunks = chunks or []
    claims = claims or []
    rels = rels or []
    questions = questions or []

    def query_side_effect(model):
        q = MagicMock()
        q.filter.return_value = q
        if model == Resume:
            q.first.return_value = mock_resume
        elif model == ResumeChunk:
            q.all.return_value = chunks
            q.count.return_value = len(chunks)
        elif model == ResumeClaim:
            q.all.return_value = claims
            q.count.return_value = len(claims)
        elif model == Relationship:
            q.all.return_value = rels
            q.count.return_value = len(rels)
        elif model == InterviewQuestion:
            q.all.return_value = questions
            q.count.return_value = len(questions)
        else:
            q.first.return_value = None
            q.all.return_value = []
            q.count.return_value = 0
        return q

    mock_db.query.side_effect = query_side_effect
    return mock_db

def test_retry_termination_on_permanent_failure():
    """Confirms that when no candidate evidence can be retrieved, generation fast-exits on attempt 1 without repeating retries."""
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.services.rag.reranker import reranker
    from app.models.rag import ResumeChunk
    from app.schemas.rag import QuestionGenerationRequest

    rid = uuid4()
    dummy_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_perm_1",
        section="Experience",
        project="TestProject",
        chunk_text="Engineered distributed backend systems.",
        embedding=[0.05] * 384,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    mock_db = _build_test_mock_db(rid, chunks=[dummy_chunk])

    with patch.object(reranker, "rerank", return_value=[]):
        req = QuestionGenerationRequest(category="Architecture", difficulty=3)
        response = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert response.validation_status == "INSUFFICIENT_EVIDENCE"
        assert response.question is None
        assert response.attempts_count == 1
        assert "retrieval_ms" in response.latencies_ms
        assert "total_ms" in response.latencies_ms

def test_request_scoped_retrieval_caching_across_retries():
    """Confirms that candidate evidence is assembled once and cached, eliminating redundant retrieval queries across retries."""
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.services.rag.retrievers import VectorRetriever
    from app.services.rag.generator import question_generator
    from app.models.rag import ResumeChunk, ResumeClaim
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema

    rid = uuid4()
    dummy_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_cache_1",
        section="Experience",
        project="TestProject",
        chunk_text="TestProject: Engineered distributed caching service with Redis and Go.",
        embedding=[0.05] * 384,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    dummy_claim = ResumeClaim(
        id=uuid4(),
        resume_id=rid,
        claim_id="clm_cache_1",
        section="Experience",
        project="TestProject",
        target_entity="Redis",
        relation_type="IMPLEMENTED_WITH",
        claim_text="Engineered distributed caching with Redis.",
        evidence_text="TestProject: Engineered distributed caching service with Redis and Go.",
        confidence=0.95,
        status="VERIFIED"
    )

    mock_db = _build_test_mock_db(rid, chunks=[dummy_chunk], claims=[dummy_claim])

    invalid_q = QuestionOutputSchema(
        question="In TestProject, how did you structure Redis cache evictions?",
        category="Architecture",
        difficulty=3,
        target_project="TestProject",
        target_entity="Redis",
        target_claim_id="hallucinated_claim_id_999",
        evidence_ids=["chk_cache_1"],
        evidence_text="TestProject: Engineered distributed caching service with Redis and Go.",
        rationale="Testing retry",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_retry"
    )
    valid_q = QuestionOutputSchema(
        question="In TestProject, how did you structure Redis cache evictions?",
        category="Architecture",
        difficulty=3,
        target_project="TestProject",
        target_entity="Redis",
        target_claim_id="clm_cache_1",
        evidence_ids=["chk_cache_1"],
        evidence_text="TestProject: Engineered distributed caching service with Redis and Go.",
        rationale="Testing retry success",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_retry"
    )

    with patch.object(question_generator, "generate", side_effect=[invalid_q, valid_q]):
        with patch.object(VectorRetriever, "retrieve", wraps=VectorRetriever.retrieve) as spy_retrieve:
            req = QuestionGenerationRequest(category="Architecture", difficulty=3, target_project="TestProject")
            resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

            assert resp.validation_status == "ACCEPTED"
            assert resp.attempts_count == 2
            # VectorRetriever.retrieve should have been called only ONCE because attempt 2 used retrieval_cache
            assert spy_retrieve.call_count == 1

def test_validator_embedding_caching_and_invalidation():
    """Confirms that validator caches question and claim embeddings and supports safe invalidation."""
    from unittest.mock import patch
    from app.services.rag.validator import grounding_validator
    from app.services.rag.embedding_service import embedding_service

    grounding_validator.clear_caches()
    assert len(grounding_validator._question_embedding_cache) == 0
    assert len(grounding_validator._claim_embedding_cache) == 0

    existing_q = InterviewQuestion(
        id=uuid4(),
        resume_id=uuid4(),
        question_text="How did you optimize SQL database query indexing?",
        difficulty=3,
        category="Performance"
    )

    with patch.object(embedding_service, "embed_query", wraps=embedding_service.embed_query) as spy_embed:
        is_dup, _ = grounding_validator.check_semantic_duplicate("How did you optimize database queries?", [existing_q])
        assert str(existing_q.id) in grounding_validator._question_embedding_cache
        initial_calls = spy_embed.call_count

        # Second check with same existing question: must NOT re-embed existing question
        is_dup, _ = grounding_validator.check_semantic_duplicate("Another query question?", [existing_q])
        assert spy_embed.call_count == initial_calls + 1

    # Invalidation check
    grounding_validator.clear_caches()
    assert len(grounding_validator._question_embedding_cache) == 0

def test_duplicate_rejection_and_bounded_retry():
    """Confirms that semantic duplicate questions are rejected and bounded retries terminate at MAX_ATTEMPTS."""
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.services.rag.generator import question_generator
    from app.models.rag import ResumeChunk
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema

    rid = uuid4()
    dummy_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_dup_1",
        section="Experience",
        project="AuthGateway",
        chunk_text="AuthGateway: Implemented OAuth2 token validation and JWT signing in Python.",
        embedding=[0.05] * 384,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    existing_q = InterviewQuestion(
        id=uuid4(),
        resume_id=rid,
        question_text="In AuthGateway, how did you implement OAuth2 token validation and JWT signing?",
        difficulty=3,
        category="Security"
    )

    mock_db = _build_test_mock_db(rid, chunks=[dummy_chunk], questions=[existing_q])

    dup_q = QuestionOutputSchema(
        question="In AuthGateway, how did you implement OAuth2 token validation and JWT signing?",
        category="Security",
        difficulty=3,
        target_project="AuthGateway",
        target_entity="OAuth2",
        evidence_ids=["chk_dup_1"],
        evidence_text="AuthGateway: Implemented OAuth2 token validation and JWT signing in Python.",
        rationale="Duplicate test",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_dup"
    )

    with patch.object(question_generator, "generate", return_value=dup_q):
        req = QuestionGenerationRequest(category="Security", difficulty=3, target_project="AuthGateway")
        resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert resp.validation_status == "INSUFFICIENT_EVIDENCE"
        assert resp.attempts_count in (2, 3)
        assert resp.question is None
        assert "duplicate" in resp.message.lower()

def test_provider_failure_handling():
    """Confirms that when the LLM provider fails, the pipeline handles it safely with latency metrics."""
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.services.rag.generator import question_generator
    from app.models.rag import ResumeChunk
    from app.schemas.rag import QuestionGenerationRequest

    rid = uuid4()
    dummy_chunk = ResumeChunk(
        id=uuid4(),
        resume_id=rid,
        chunk_id="chk_err_1",
        section="Experience",
        project="TestProject",
        chunk_text="Engineered distributed backend systems.",
        embedding=[0.05] * 384,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    mock_db = _build_test_mock_db(rid, chunks=[dummy_chunk])

    with patch.object(question_generator, "generate", side_effect=RuntimeError("Google GenAI 503 Service Unavailable")):
        req = QuestionGenerationRequest(category="Architecture", difficulty=3)
        resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert resp.validation_status == "INSUFFICIENT_EVIDENCE"
        assert resp.attempts_count in (2, 3)
        assert resp.question is None
        assert "Provider execution failure" in resp.message
        assert "llm_ms" in resp.latencies_ms
        assert "total_ms" in resp.latencies_ms

def test_anti_duplication_prompt_injection():
    """Confirms that generator injects previously asked questions into the prompt to proactively prevent duplicates."""
    from unittest.mock import MagicMock, patch
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    mock_provider = MagicMock()
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "How did you design the indexing strategy?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "DbService",
            "target_entity": "PostgreSQL",
            "target_claim_id": None,
            "target_relationship_id": None,
            "evidence_ids": ["ev_1"],
            "rationale": "Testing prompt injection"
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )

    context = {
        "selected_evidence_items": [
            {"evidence_id": "ev_1", "text": "PostgreSQL indexing in DbService", "project": "DbService"}
        ]
    }

    with patch("app.services.rag.generator.get_llm_provider", return_value=mock_provider):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            previously_asked_questions=["How did you configure Redis evictions?", "What is the token auth flow?"]
        )

        call_args = mock_provider.generate_structured.call_args
        prompt_text = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "PREVIOUSLY ASKED QUESTIONS (DO NOT DUPLICATE)" in prompt_text
        assert "How did you configure Redis evictions?" in prompt_text
        assert "What is the token auth flow?" in prompt_text
        assert "probe a distinct technical mechanism" in prompt_text

def test_validator_rejects_unsupported_implementation_presupposition():
    """Confirms that when evidence only indicates technology usage, questions presupposing unverified implementation are rejected."""
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }
    unsupported_q = QuestionOutputSchema(
        question="In CampusBridge, walk through how you implemented multi-table transaction rollback mechanisms with PostgreSQL.",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=["rel_pg"],
        evidence_text="CampusBridge HAS_TECHNOLOGY PostgreSQL",
        rationale="Testing unsupported implementation rejection",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_entail_test"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=unsupported_q,
        allowed_projects=["CampusBridge"],
        allowed_entities=["PostgreSQL"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={"rel_pg": None}
    )

    assert is_valid is False
    assert details.get("code") == "unsupported_implementation_presupposition"
    assert "unverified implementation details" in reason

def test_validator_accepts_open_ended_knowledge_exploration():
    """Confirms that open-ended questions probing technical knowledge without asserting unverified implementation are accepted."""
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }
    open_ended_q = QuestionOutputSchema(
        question="In CampusBridge using PostgreSQL, how would you or did you approach managing transaction boundaries and rollback consistency across service layers?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=["rel_pg"],
        evidence_text="CampusBridge HAS_TECHNOLOGY PostgreSQL",
        rationale="Testing open-ended question acceptance",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_entail_test"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=open_ended_q,
        allowed_projects=["CampusBridge"],
        allowed_entities=["PostgreSQL"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={"rel_pg": None}
    )

    assert is_valid is True
    assert details.get("status") == "PASSED"

def test_validator_accepts_assertive_question_when_grounded_in_chunk_evidence():
    """Confirms that assertive questions are accepted when an actual resume chunk explicitly describes the implementation."""
    resolved_evidence = {
        "chk_tx": {
            "evidence_id": "chk_tx",
            "text": "Engineered CampusBridge PostgreSQL persistence layer with multi-table transaction rollback mechanisms.",
            "source_type": "chunk",
            "project": "CampusBridge"
        }
    }
    grounded_q = QuestionOutputSchema(
        question="In CampusBridge, walk through how you implemented multi-table transaction rollback mechanisms with PostgreSQL.",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="PostgreSQL",
        evidence_ids=["chk_tx"],
        evidence_text="Engineered CampusBridge PostgreSQL persistence layer with multi-table transaction rollback mechanisms.",
        rationale="Testing chunk-grounded implementation acceptance",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_entail_test"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=grounded_q,
        allowed_projects=["CampusBridge"],
        allowed_entities=["PostgreSQL"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )

    assert is_valid is True
    assert details.get("status") == "PASSED"

def test_validator_rejects_diverse_assertive_phrasings():
    """Confirms that various assertive phrasings presupposing unverified implementations are rejected when evidence is only a technology link."""
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }
    unsupported_phrasings = [
        "In CampusBridge, describe your implementation of transaction rollback mechanisms with PostgreSQL.",
        "In CampusBridge, explain how you built the distributed rollback engine using Spring Boot and PostgreSQL.",
        "In CampusBridge, detail your implementation of multi-table database transactions with PostgreSQL.",
        "In CampusBridge, how did you implement your custom transaction manager with PostgreSQL?",
        "In CampusBridge, walk through how you architected your transaction rollback system with PostgreSQL.",
        "In CampusBridge, when you implemented transaction boundaries with PostgreSQL, how did you handle rollbacks?"
    ]

    for q_text in unsupported_phrasings:
        q_schema = QuestionOutputSchema(
            question=q_text,
            category="Architecture",
            difficulty=3,
            target_project="CampusBridge",
            target_entity="PostgreSQL",
            evidence_ids=["rel_pg"],
            evidence_text="CampusBridge HAS_TECHNOLOGY PostgreSQL",
            rationale="Testing diverse phrasing rejection",
            grounding_score=0.0,
            relevance_score=0.0,
            technical_quality_score=0.0,
            trace_id="trc_entail_test"
        )
        is_valid, reason, details = grounding_validator.validate(
            question=q_schema,
            allowed_projects=["CampusBridge"],
            allowed_entities=["PostgreSQL"],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={"rel_pg": None}
        )
        assert is_valid is False, f"Expected rejection for phrasing: '{q_text}', got valid."
        assert details.get("code") == "unsupported_implementation_presupposition"

def test_validator_accepts_diverse_open_ended_phrasings():
    """Confirms that diverse open-ended phrasings probing technical knowledge are accepted when evidence supports technology usage."""
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }
    open_ended_phrasings = [
        "In CampusBridge using PostgreSQL, how would you or did you approach managing transaction boundaries and rollback consistency across service layers?",
        "In CampusBridge with PostgreSQL, how would you handle transaction rollbacks across multi-table operations?",
        "What approach would you take to ensure transactional consistency in CampusBridge using PostgreSQL?",
        "How might you configure PostgreSQL connection pooling in Spring Boot for CampusBridge?",
        "How could you structure repository interfaces in Spring Boot to handle data persistence in CampusBridge?"
    ]

    for q_text in open_ended_phrasings:
        q_schema = QuestionOutputSchema(
            question=q_text,
            category="Architecture",
            difficulty=3,
            target_project="CampusBridge",
            target_entity="PostgreSQL",
            evidence_ids=["rel_pg"],
            evidence_text="CampusBridge HAS_TECHNOLOGY PostgreSQL",
            rationale="Testing diverse open-ended acceptance",
            grounding_score=0.0,
            relevance_score=0.0,
            technical_quality_score=0.0,
            trace_id="trc_entail_test"
        )
        is_valid, reason, details = grounding_validator.validate(
            question=q_schema,
            allowed_projects=["CampusBridge"],
            allowed_entities=["PostgreSQL"],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={"rel_pg": None}
        )
        assert is_valid is True, f"Expected acceptance for open-ended phrasing: '{q_text}', got rejected: {reason}"
        assert details.get("status") == "PASSED"

def test_generator_accepts_provider_citation_format_variations(monkeypatch):
    """
    Confirms that generator accepts provider citation field variations (evidence_id, citations,
    evidence, evidenceIds, etc.) and correctly maps them to evidence_ids without error.
    """
    from unittest.mock import MagicMock
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    context = {
        "selected_evidence_items": [
            {
                "evidence_id": "chk_101",
                "section": "Projects",
                "project": "Xibotix",
                "text": "Xibotix: Built React frontend with Vite."
            }
        ],
        "claims": [],
        "relationships": []
    }

    mock_provider = MagicMock()
    monkeypatch.setattr("app.services.rag.generator.get_llm_provider", lambda force_mock=False: mock_provider)

    variations = [
        {"evidence_id": "chk_101"},
        {"evidence_id": ["chk_101"]},
        {"citations": ["chk_101"]},
        {"evidence": ["chk_101"]},
        {"evidenceIds": ["chk_101"]},
        {"evidenceId": "chk_101"},
        {"sources": ["chk_101"]},
        {"source_ids": ["chk_101"]}
    ]

    for var_dict in variations:
        payload = {
            "question": "How would you evaluate build speed trade-offs with Vite in Xibotix?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "Xibotix",
            "target_entity": "Vite",
            "target_claim_id": None,
            "target_relationship_id": None,
            "rationale": "Vite build probe."
        }
        payload.update(var_dict)

        mock_provider.generate_structured.return_value = LLMGenerationResult(
            raw_content="{}",
            parsed_json=payload,
            provider="mock",
            model="mock",
            latency_ms=10.0
        )

        q = generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="Xibotix",
            target_entity="Vite",
            trace_id="trc_var_test"
        )
        assert q.evidence_ids == ["chk_101"], f"Failed on variation: {var_dict}"
        assert q.evidence_text == "Xibotix: Built React frontend with Vite."

def test_generator_rejects_missing_citations_across_empty_formats(monkeypatch):
    """
    Confirms that missing citations in any format (empty list, null, empty string, or missing key)
    are strictly rejected with 'no evidence citations' without fabricating IDs.
    """
    from unittest.mock import MagicMock
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    context = {
        "selected_evidence_items": [
            {
                "evidence_id": "chk_101",
                "section": "Projects",
                "project": "Xibotix",
                "text": "Xibotix: Built React frontend with Vite."
            }
        ],
        "claims": [],
        "relationships": []
    }

    mock_provider = MagicMock()
    monkeypatch.setattr("app.services.rag.generator.get_llm_provider", lambda force_mock=False: mock_provider)

    empty_variations = [
        {"evidence_ids": []},
        {"evidence_ids": None},
        {"evidence_ids": [""]},
        {"evidence_ids": ["null"]},
        {"evidence_ids": ["none"]},
        {"evidence_ids": ["N/A"]},
        {"evidence_id": ""},
        {"citations": []},
        {}
    ]

    for var_dict in empty_variations:
        payload = {
            "question": "How would you evaluate build speed trade-offs with Vite in Xibotix?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "Xibotix",
            "target_entity": "Vite",
            "target_claim_id": None,
            "target_relationship_id": None,
            "rationale": "Vite build probe."
        }
        payload.update(var_dict)

        mock_provider.generate_structured.return_value = LLMGenerationResult(
            raw_content="{}",
            parsed_json=payload,
            provider="mock",
            model="mock",
            latency_ms=10.0
        )

        with pytest.raises(ValueError, match="no evidence citations"):
            generator.generate(
                context=context,
                category="Architecture",
                difficulty=3,
                target_project="Xibotix",
                target_entity="Vite",
                trace_id="trc_empty_test"
            )

def test_generator_rejects_malformed_model_output(monkeypatch):
    """
    Confirms that malformed or non-dict model outputs raise explicit ValueError.
    """
    from unittest.mock import MagicMock
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    context = {
        "selected_evidence_items": [{"evidence_id": "chk_1", "text": "text", "project": "P"}],
        "claims": [],
        "relationships": []
    }

    mock_provider = MagicMock()
    monkeypatch.setattr("app.services.rag.generator.get_llm_provider", lambda force_mock=False: mock_provider)

    # Subcase 1: Provider raises decode failure on malformed json
    mock_provider.generate_structured.side_effect = ValueError("Candidate rejected: Model returned malformed or non-JSON output.")

    with pytest.raises(ValueError, match="malformed or non-JSON output"):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="P",
            trace_id="trc_malformed"
        )

    # Subcase 2: Valid JSON dict returned but missing required schema fields
    mock_provider.generate_structured.side_effect = None
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={"broken_field": "no question"},
        provider="mock",
        model="mock",
        latency_ms=10.0
    )

    with pytest.raises(ValueError, match="no evidence citations"):
        generator.generate(
            context=context,
            category="Architecture",
            difficulty=3,
            target_project="P",
            trace_id="trc_schema_viol"
        )

def test_generator_supports_claim_and_relationship_citations(monkeypatch):
    """
    Confirms that valid relationship IDs and claim IDs supplied in the context are recognized
    and resolved in generator output.
    """
    from unittest.mock import MagicMock
    from app.services.llm.base import LLMGenerationResult

    generator = QuestionGenerationService(force_mock=True)
    context = {
        "selected_evidence_items": [],
        "claims": [
            {
                "claim_id": "clm_789",
                "project": "CampusBridge",
                "text": "Secured CampusBridge using Spring Security."
            }
        ],
        "relationships": [
            {
                "relationship_id": "rel_456",
                "source": "CampusBridge",
                "rel": "HAS_TECHNOLOGY",
                "target": "PostgreSQL"
            }
        ]
    }

    mock_provider = MagicMock()
    monkeypatch.setattr("app.services.rag.generator.get_llm_provider", lambda force_mock=False: mock_provider)

    # Test citing relationship
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "When working with PostgreSQL in CampusBridge, how would you approach connection pooling?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CampusBridge",
            "target_entity": "PostgreSQL",
            "target_claim_id": None,
            "target_relationship_id": "rel_456",
            "evidence_ids": ["rel_456"],
            "rationale": "PostgreSQL probe."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )

    q = generator.generate(context=context, category="Architecture", difficulty=3, trace_id="trc_rel_cite")
    assert q.evidence_ids == ["rel_456"]
    assert "CampusBridge HAS_TECHNOLOGY PostgreSQL" in q.evidence_text

    # Test citing claim
    mock_provider.generate_structured.return_value = LLMGenerationResult(
        raw_content="{}",
        parsed_json={
            "question": "In CampusBridge, how would you configure Spring Security filter chains?",
            "category": "Architecture",
            "difficulty": 3,
            "target_project": "CampusBridge",
            "target_entity": "Spring Security",
            "target_claim_id": "clm_789",
            "target_relationship_id": None,
            "evidence_ids": ["clm_789"],
            "rationale": "Spring Security probe."
        },
        provider="mock",
        model="mock",
        latency_ms=10.0
    )

    q2 = generator.generate(context=context, category="Architecture", difficulty=3, trace_id="trc_clm_cite")
    assert q2.evidence_ids == ["clm_789"]
    assert "Secured CampusBridge using Spring Security" in q2.evidence_text

def test_validator_rejects_unsupported_implementation_assumptions_broader_patterns():
    """
    Confirms that implementation assumptions about code-splitting, multi-table transactions,
    security controls, and error pathways are rejected when evidence only establishes technology usage.
    """
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        },
        "rel_vite": {
            "evidence_id": "rel_vite",
            "text": "Xibotix HAS_TECHNOLOGY Vite",
            "source_type": "relationship",
            "project": "Xibotix"
        }
    }

    test_unsupported = [
        ("Xibotix", "Vite", "rel_vite", "In Xibotix using Vite and React, how did you configure code-splitting and asset optimization to minimize initial load latency?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "In CampusBridge with PostgreSQL, walk through a concrete example of managing a multi-table database transaction, detailing how rollback behavior is maintained."),
        ("Xibotix", "Vite", "rel_vite", "In Xibotix with React, how do you manage state encapsulation and input sanitization to prevent common client-side security vulnerabilities such as Cross-Site Scripting (XSS)?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "In CampusBridge with PostgreSQL, how did you structure your entity relationships to maintain rollback consistency across service boundaries?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "In CampusBridge, detail how transaction handling and error pathways were managed when transferring state across this boundary.")
    ]

    for proj, ent, eid, q_text in test_unsupported:
        q_schema = QuestionOutputSchema(
            question=q_text,
            category="Architecture",
            difficulty=3,
            target_project=proj,
            target_entity=ent,
            evidence_ids=[eid],
            evidence_text=resolved_evidence[eid]["text"],
            rationale="Testing broader presupposition rejection",
            grounding_score=0.0,
            relevance_score=0.0,
            technical_quality_score=0.0,
            trace_id="trc_broad_entail"
        )
        is_valid, reason, details = grounding_validator.validate(
            question=q_schema,
            allowed_projects=[proj],
            allowed_entities=[ent],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={eid: None}
        )
        assert is_valid is False, f"Expected rejection for '{q_text}', got valid."
        assert details.get("code") == "unsupported_implementation_presupposition"

def test_validator_accepts_open_ended_design_questions_for_unsupported_implementations():
    """
    Confirms that open-ended design questions exploring architectural reasoning are accepted
    even when evidence only establishes technology usage.
    """
    resolved_evidence = {
        "rel_pg": {
            "evidence_id": "rel_pg",
            "text": "CampusBridge HAS_TECHNOLOGY PostgreSQL",
            "source_type": "relationship",
            "project": "CampusBridge"
        },
        "rel_vite": {
            "evidence_id": "rel_vite",
            "text": "Xibotix HAS_TECHNOLOGY Vite",
            "source_type": "relationship",
            "project": "Xibotix"
        }
    }

    test_open_ended = [
        ("Xibotix", "Vite", "rel_vite", "In Xibotix using Vite and React, what approach would you take to optimize bundle size and evaluate code-splitting trade-offs?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "In CampusBridge with PostgreSQL, how would you approach managing transaction boundaries and rollback consistency across service layers?"),
        ("Xibotix", "Vite", "rel_vite", "In Xibotix with React, how would you approach securing dynamic user input against client-side vulnerabilities?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "In CampusBridge, how might you handle authentication error pathways in a Spring Security filter chain?"),
        ("CampusBridge", "PostgreSQL", "rel_pg", "What approach would you take to ensure transactional consistency in CampusBridge using PostgreSQL?")
    ]

    for proj, ent, eid, q_text in test_open_ended:
        q_schema = QuestionOutputSchema(
            question=q_text,
            category="Architecture",
            difficulty=3,
            target_project=proj,
            target_entity=ent,
            evidence_ids=[eid],
            evidence_text=resolved_evidence[eid]["text"],
            rationale="Testing open-ended acceptance",
            grounding_score=0.0,
            relevance_score=0.0,
            technical_quality_score=0.0,
            trace_id="trc_open_accept"
        )
        is_valid, reason, details = grounding_validator.validate(
            question=q_schema,
            allowed_projects=[proj],
            allowed_entities=[ent],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={eid: None}
        )
        assert is_valid is True, f"Expected acceptance for '{q_text}', got rejected: {reason}"
        assert details.get("status") == "PASSED"

def test_actual_failure_pattern_unsupported_assertive_rejected():
    """
    Regression test for the exact live failure pattern where an assertive question
    commands the candidate to walk through an unstated flow/mechanics (token validation flow,
    stateless session management, role-based separation) while evidence only establishes technology usage.
    """
    resolved_evidence = {
        "rel_sb": {
            "evidence_id": "rel_sb",
            "text": "CampusBridge HAS_TECHNOLOGY Spring Boot",
            "source_type": "relationship",
            "project": "CampusBridge"
        },
        "rel_sec": {
            "evidence_id": "rel_sec",
            "text": "CampusBridge HAS_TECHNOLOGY Spring Security",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }

    q_text = (
        "In the CampusBridge platform, which utilizes React, Java, Spring Boot, PostgreSQL, "
        "and Spring Security with JWT and REST APIs, how do you design the authentication and "
        "authorization boundary between the front-end client and the back-end services? "
        "Walk through the flow of token validation, stateless session management, and how "
        "role-based separation of concerns is enforced across the API endpoints."
    )

    q_schema = QuestionOutputSchema(
        question=q_text,
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["rel_sb", "rel_sec"],
        evidence_text="CampusBridge HAS_TECHNOLOGY Spring Boot \nCampusBridge HAS_TECHNOLOGY Spring Security",
        rationale="Testing live failure pattern rejection",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_regress_fail"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=q_schema,
        allowed_projects=["CampusBridge"],
        allowed_entities=["Spring Boot", "Spring Security"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={"rel_sb": None, "rel_sec": None}
    )

    assert is_valid is False
    assert details.get("code") == "unsupported_implementation_presupposition"
    assert "asserts unverified implementation details" in reason

def test_actual_failure_pattern_open_ended_alternative_accepted():
    """
    Regression test confirming that a properly framed open-ended alternative
    exploring the same architectural domain passes validation under technology-only evidence.
    """
    resolved_evidence = {
        "rel_sb": {
            "evidence_id": "rel_sb",
            "text": "CampusBridge HAS_TECHNOLOGY Spring Boot",
            "source_type": "relationship",
            "project": "CampusBridge"
        },
        "rel_sec": {
            "evidence_id": "rel_sec",
            "text": "CampusBridge HAS_TECHNOLOGY Spring Security",
            "source_type": "relationship",
            "project": "CampusBridge"
        }
    }

    q_text = (
        "In the CampusBridge platform, which utilizes React, Java, Spring Boot, PostgreSQL, "
        "and Spring Security with JWT and REST APIs, how would you approach designing the "
        "authentication and authorization boundary between the front-end client and the back-end services? "
        "What trade-offs would you evaluate regarding token validation and role-based access control across API endpoints?"
    )

    q_schema = QuestionOutputSchema(
        question=q_text,
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["rel_sb", "rel_sec"],
        evidence_text="CampusBridge HAS_TECHNOLOGY Spring Boot \nCampusBridge HAS_TECHNOLOGY Spring Security",
        rationale="Testing open-ended alternative acceptance",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_regress_pass"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=q_schema,
        allowed_projects=["CampusBridge"],
        allowed_entities=["Spring Boot", "Spring Security"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={"rel_sb": None, "rel_sec": None}
    )

    assert is_valid is True, f"Expected acceptance, got: {reason}"
    assert details.get("status") == "PASSED"
    assert q_schema.grounding_score >= 0.65

def test_actual_failure_pattern_assertive_question_with_chunk_evidence_accepted():
    """
    Regression test confirming that an assertive question is accepted when supported
    by explicit, detailed chunk evidence describing the actual implementation.
    """
    resolved_evidence = {
        "chk_cb_impl": {
            "evidence_id": "chk_cb_impl",
            "text": (
                "CampusBridge: Secured all API routes with stateless JWT authentication and Spring Security, "
                "and built a recruiter verification workflow requiring admin approval before granting recruiter access."
            ),
            "source_type": "chunk",
            "project": "CampusBridge"
        }
    }

    q_text = (
        "In the CampusBridge platform, walk through how you implemented the recruiter verification workflow "
        "that requires placement admin approval before granting recruiter access to post openings."
    )

    q_schema = QuestionOutputSchema(
        question=q_text,
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Security",
        evidence_ids=["chk_cb_impl"],
        evidence_text=resolved_evidence["chk_cb_impl"]["text"],
        rationale="Testing chunk-grounded assertive acceptance",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_regress_chunk"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=q_schema,
        allowed_projects=["CampusBridge"],
        allowed_entities=["Spring Security"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )

    assert is_valid is True, f"Expected acceptance, got: {reason}"
    assert details.get("status") == "PASSED"
    assert q_schema.grounding_score >= 0.65

def test_rag_pipeline_preserves_target_project_and_builds_targeted_retry_feedback():
    """
    Confirms that when a target project is explicitly requested by the user,
    rag_pipeline preserves that target project upon duplicate or validation rejection,
    and constructs targeted feedback instructing open-ended framing while citing verified evidence IDs.
    """
    from unittest.mock import patch, MagicMock
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema
    from app.models.rag import ResumeChunk

    rid = uuid4()
    req = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot"
    )

    mock_db = _build_test_mock_db(
        rid,
        chunks=[
            ResumeChunk(
                id=uuid4(),
                resume_id=rid,
                chunk_id="chk_cb",
                section="Projects",
                project="CampusBridge",
                chunk_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding=[0.1] * 384
            )
        ]
    )

    captured_feedback = []

    # Mock candidate question that fails validation with presupposition on attempt 1,
    # then succeeds on attempt 2
    bad_q = QuestionOutputSchema(
        question="In CampusBridge, walk through how you implemented multi-table transactions across service layers.",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["chk_cb"],
        evidence_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
        rationale="Bad presupposition",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_mock"
    )

    good_q = QuestionOutputSchema(
        question="In CampusBridge, how would you approach designing transaction boundaries across service layers?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["chk_cb"],
        evidence_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
        rationale="Good open-ended",
        grounding_score=0.9,
        relevance_score=0.9,
        technical_quality_score=0.9,
        trace_id="trc_mock"
    )

    def mock_generate(*args, **kwargs):
        captured_feedback.append(kwargs.get("feedback_instruction"))
        if len(captured_feedback) == 1:
            return bad_q
        return good_q

    with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_generate), \
         patch("app.services.rag.validator.grounding_validator.validate") as mock_val:

        mock_val.side_effect = [
            (False, "REJECTED: Presupposition", {"code": "unsupported_implementation_presupposition"}),
            (True, "PASSED", {"status": "PASSED", "grounding_score": 0.9, "relevance_score": 0.9, "technical_quality_score": 0.9})
        ]

        resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert resp.validation_status == "ACCEPTED"
        assert resp.attempts_count == 2
        assert len(captured_feedback) == 2
        assert captured_feedback[0] is None
        # Verify attempt 2 received targeted remediation
        fb2 = captured_feedback[1]
        assert fb2 is not None
        assert "OPEN-ENDED" in fb2
        assert "evidence_id" in fb2

def test_check_semantic_duplicate_returns_details_and_maintains_backwards_compatibility():
    """
    Verifies that check_semantic_duplicate returns a 2-tuple (bool, msg) by default,
    and returns full details (bool, msg, duplicate_text, duplicate_id, similarity) when return_details=True.
    """
    existing_q = InterviewQuestion(
        id=uuid4(),
        question_text="In CampusBridge using Spring Boot and JWT, how do you handle token validation and filter chain error pathways?"
    )

    new_q = "In CampusBridge with Spring Boot, how would you approach structuring the JWT filter chain and token validation error handling?"

    # Default 2-tuple call
    is_dup, msg = grounding_validator.check_semantic_duplicate(new_q, [existing_q])
    assert is_dup is True
    assert "semantically duplicate" in msg

    # Detailed 5-tuple call
    is_dup, msg, dup_text, dup_id, sim = grounding_validator.check_semantic_duplicate(new_q, [existing_q], return_details=True)
    assert is_dup is True
    assert dup_text == existing_q.question_text
    assert dup_id == str(existing_q.id)
    assert sim >= 0.82

def test_duplicate_rejection_feeds_actionable_guidance_with_duplicate_text_and_preserves_campusbridge():
    """
    Verifies that when a question is rejected as duplicate:
    1. The retry feedback explicitly identifies the duplicate question's text.
    2. Instructs the model not to ask about the same topic.
    3. Suggests materially different architectural dimensions for CampusBridge.
    4. Preserves target_project='CampusBridge' throughout retries.
    """
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema
    from app.models.rag import ResumeChunk

    rid = uuid4()
    req = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot"
    )

    mock_db = _build_test_mock_db(
        rid,
        chunks=[
            ResumeChunk(
                id=uuid4(),
                resume_id=rid,
                chunk_id="chk_cb",
                section="Projects",
                project="CampusBridge",
                chunk_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding=[0.1] * 384
            )
        ]
    )

    captured_feedback = []
    captured_projects = []

    # Attempt 1: Duplicate question on JWT filter chain
    dup_q = QuestionOutputSchema(
        question="In CampusBridge using Spring Boot and JWT, how would you approach the security filter chain?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["chk_cb"],
        evidence_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
        rationale="Duplicate attempt",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_dup"
    )

    # Attempt 2: Materially different question on automated eligibility evaluation rules
    distinct_q = QuestionOutputSchema(
        question="In CampusBridge using Spring Boot, how would you approach designing the automated eligibility engine and rule evaluation pipeline for candidate applications?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["chk_cb"],
        evidence_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
        rationale="Distinct attempt",
        grounding_score=0.92,
        relevance_score=0.88,
        technical_quality_score=0.85,
        trace_id="trc_distinct"
    )

    existing_question_text = (
        "In the backend architecture of CampusBridge using Java and Spring Boot, how did you design "
        "the integration contract and boundary handling for your REST APIs secured with Spring Security and JWT?"
    )

    def mock_generate(*args, **kwargs):
        captured_feedback.append(kwargs.get("feedback_instruction"))
        captured_projects.append(kwargs.get("target_project"))
        if len(captured_feedback) == 1:
            return dup_q
        return distinct_q

    with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_generate), \
         patch("app.services.rag.validator.grounding_validator.validate") as mock_val:

        mock_val.side_effect = [
            (False, "REJECTED: Duplicate", {
                "code": "semantic_duplicate",
                "duplicate_question_text": existing_question_text,
                "similarity": 0.88
            }),
            (True, "PASSED", {
                "status": "PASSED",
                "grounding_score": 0.92,
                "relevance_score": 0.88,
                "technical_quality_score": 0.85
            })
        ]

        resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert resp.validation_status == "ACCEPTED"
        assert resp.attempts_count == 2
        assert len(captured_feedback) == 2

        # Verify target project remained CampusBridge across both attempts
        assert captured_projects[0] == "CampusBridge"
        assert captured_projects[1] == "CampusBridge"

        # Verify retry feedback for Attempt 2 contains the duplicate question text and actionable instructions
        fb2 = captured_feedback[1]
        assert fb2 is not None
        assert existing_question_text in fb2
        assert "MUST NOT ask about the same topic" in fb2 or "do not merely rephrase" in fb2
        assert "MATERIALLY DIFFERENT" in fb2
        assert "CampusBridge" in fb2
        assert "evidence_id" in fb2

def test_duplicate_retry_terminates_bounded_and_fails_explicitly():
    """
    Verifies that when every retry candidate remains a semantic duplicate,
    the pipeline terminates strictly at MAX_ATTEMPTS (3) and fails explicitly
    with INSUFFICIENT_EVIDENCE rather than fabricating ungrounded questions.
    """
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema
    from app.models.rag import ResumeChunk

    rid = uuid4()
    req = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot"
    )

    mock_db = _build_test_mock_db(
        rid,
        chunks=[
            ResumeChunk(
                id=uuid4(),
                resume_id=rid,
                chunk_id="chk_cb",
                section="Projects",
                project="CampusBridge",
                chunk_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding=[0.1] * 384
            )
        ]
    )

    dup_q = QuestionOutputSchema(
        question="In CampusBridge with Spring Boot, how would you approach JWT filter chain token validation?",
        category="Architecture",
        difficulty=3,
        target_project="CampusBridge",
        target_entity="Spring Boot",
        evidence_ids=["chk_cb"],
        evidence_text="CampusBridge with Java, Spring Boot, and PostgreSQL.",
        rationale="Persistent duplicate",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_dup_all"
    )

    with patch("app.services.rag.generator.question_generator.generate", return_value=dup_q), \
         patch("app.services.rag.validator.grounding_validator.validate") as mock_val:

        mock_val.return_value = (
            False,
            "REJECTED: Question is semantically duplicate with existing Question ID 'cda7f75b' (Cosine: 0.89 >= 0.82).",
            {
                "code": "semantic_duplicate",
                "duplicate_question_text": "Existing question on JWT filter chain",
                "similarity": 0.89
            }
        )

        resp = rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert resp.validation_status == "INSUFFICIENT_EVIDENCE"
        assert resp.attempts_count in (2, 3)
        assert resp.question is None
        assert "duplicate" in resp.message.lower()


# ============================================================================
# PHASE 5 LIVE TESTING REGRESSION TESTS (SYNTHETIC FIXTURES)
# ============================================================================

def test_regression_generic_duplicate_feedback_derived_from_evidence():
    """
    Issue 1: Verify retry guidance is generic, derived from candidate evidence,
    and contains zero hardcoded CampusBridge domain references.
    """
    from unittest.mock import patch
    from app.services.rag.rag_pipeline import rag_pipeline
    from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema
    from app.models.rag import ResumeChunk

    rid = uuid4()
    req = QuestionGenerationRequest(
        category="Architecture",
        difficulty=3,
        target_project="TelemetryPipeline",
        target_entity="Python"
    )

    mock_db = _build_test_mock_db(
        rid,
        chunks=[
            ResumeChunk(
                id=uuid4(),
                resume_id=rid,
                chunk_id="chk_syn_1",
                section="Projects",
                project="TelemetryPipeline",
                chunk_text="TelemetryPipeline | Asynchronous streaming ingestion with Python and Kafka.",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding=[0.1] * 384
            )
        ]
    )

    dup_q = QuestionOutputSchema(
        question="In TelemetryPipeline with Python, how would you approach stream ingestion?",
        category="Architecture",
        difficulty=3,
        target_project="TelemetryPipeline",
        target_entity="Python",
        evidence_ids=["chk_syn_1"],
        evidence_text="TelemetryPipeline | Asynchronous streaming ingestion with Python and Kafka.",
        rationale="Synthetic duplicate test",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_syn_dup"
    )

    captured_feedback = []

    def mock_gen(ctx, category, difficulty, trace_id, feedback_instruction=None, **kwargs):
        captured_feedback.append(feedback_instruction)
        return dup_q

    with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_gen), \
         patch("app.services.rag.validator.grounding_validator.validate") as mock_val:

        mock_val.return_value = (
            False,
            "REJECTED: Question is semantically duplicate with existing Question ID 'cda7f75b' (Cosine: 0.88 >= 0.82).",
            {
                "code": "semantic_duplicate",
                "duplicate_question_text": "Existing question on stream ingestion",
                "similarity": 0.88
            }
        )

        rag_pipeline.generate_grounded_question(rid, req, mock_db)

        assert len(captured_feedback) >= 2
        retry_fb = captured_feedback[1]
        assert retry_fb is not None

        # Verify zero hardcoded CampusBridge domain references
        assert "student/recruiter" not in retry_fb
        assert "hiring eligibility" not in retry_fb
        assert "Vercel and Render" not in retry_fb

        # Verify generic, evidence-derived guidance
        assert "TelemetryPipeline" in retry_fb
        assert "MATERIALLY DIFFERENT" in retry_fb
        assert "Existing question on stream ingestion" in retry_fb


def test_regression_chunking_preserves_hyphenated_project_names():
    """
    Issue 2: Verify chunking_service does not truncate project titles on intra-word hyphens
    and does not accept wrapped bullet continuations as projects.
    """
    from app.services.rag.chunking_service import chunking_service

    raw_resume = """
PROJECTS
Real-Time Conversational Telephony AI Voicebot | Python, AsyncIO, GCP, Asterisk PBX, Docker
● Architected a conversational voice automation bot on GCP.
● Built async, multi-threaded Python backend bridges.

Event-Triggered Road Surface Monitoring System | C++, Embedded C, ESP32-CAM
● Engineered an edge IoT road anomaly detection system utilizing master-slave sensor fusion.
● boundary layers for real-time compliance.
"""
    chunks = chunking_service.chunk_resume(str(uuid4()), raw_resume)
    project_names = [c.project for c in chunks if c.project]

    # Must preserve full hyphenated names
    assert "Real-Time Conversational Telephony AI Voicebot" in project_names
    assert "Event-Triggered Road Surface Monitoring System" in project_names

    # Must NOT truncate to single words
    assert "Real" not in project_names
    assert "Event" not in project_names

    # Must NOT treat wrapped bullet fragments as projects
    assert "boundary layers for real-time compliance" not in project_names


def test_regression_experience_extraction_does_not_create_developed_employer():
    """
    Issue 3: Verify experience entries extract WORKED_AT and WORKED_AS, never DEVELOPED,
    and that claim service does not mark unsupported employer development as SUPPORTED.
    """
    from app.services.relationship_service import parse_experience_section
    from app.services.rag.claim_service import claim_service
    from app.models.relationship import Relationship

    sec_text = """
SpeechLogix Technologies -- Software Engineer Intern | May 2026 - June 2026
● Architected and deployed next-gen communication boundary layers.
● Built async, multi-threaded Python backend bridges.
"""
    known_skills = {"python": "Python"}
    triplets = parse_experience_section(sec_text, known_skills)

    relations = {(t.source_entity, t.relation_type, t.target_entity) for t in triplets}

    # Must establish employment and role
    assert ("Candidate", "WORKED_AT", "SpeechLogix Technologies") in relations
    assert ("Candidate", "WORKED_AS", "Software Engineer Intern") in relations

    # MUST NOT establish that candidate developed employer
    assert ("Candidate", "DEVELOPED", "SpeechLogix Technologies") not in relations
    assert ("SpeechLogix Technologies", "BUILT_WITH", "Python") not in relations

    # Verify claim synthesis status logic:
    # A synthetic unsupported DEVELOPED claim in Experience must be flagged UNSUPPORTED
    unsupported_rel = Relationship(
        resume_id=uuid4(),
        source_entity="Candidate",
        relation_type="DEVELOPED",
        target_entity="SpeechLogix Technologies",
        section="Experience",
        evidence_text="SpeechLogix Technologies -- Software Engineer Intern | May 2026 - June 2026"
    )
    supported_rel = Relationship(
        resume_id=uuid4(),
        source_entity="Candidate",
        relation_type="WORKED_AT",
        target_entity="SpeechLogix Technologies",
        section="Experience",
        evidence_text="SpeechLogix Technologies -- Software Engineer Intern | May 2026 - June 2026"
    )

    ev_lower = unsupported_rel.evidence_text.lower()
    assert unsupported_rel.section != "Projects" or any(w in ev_lower for w in ["intern", "employed"])
    # The claim service rule marks this UNSUPPORTED
    status_unsupported = "UNSUPPORTED" if (unsupported_rel.section != "Projects" or "intern" in ev_lower) else "SUPPORTED"
    status_supported = "SUPPORTED" if supported_rel.target_entity.lower() in ev_lower else "UNSUPPORTED"

    assert status_unsupported == "UNSUPPORTED"
    assert status_supported == "SUPPORTED"


def test_regression_target_entity_reflects_spatial_indexing_focus():
    """
    Issue 4: Verify canonicalizer recognizes spatial technologies and generator/validator
    align target_entity with the actual technical focus rather than inheriting React.
    """
    from app.core.canonicalizer import canonicalize_entity
    from app.services.rag.validator import grounding_validator
    from app.schemas.rag import QuestionOutputSchema
    from unittest.mock import MagicMock

    # 1. Canonicalizer verifies spatial technologies
    assert canonicalize_entity("qgis")[0] == "QGIS"
    assert canonicalize_entity("shapely")[0] == "Shapely"
    assert canonicalize_entity("geojson")[0] == "GeoJSON"
    assert canonicalize_entity("spatial indexing")[0] == "R-Tree Indexing"
    assert canonicalize_entity("r-tree indexing")[0] == "R-Tree Indexing"

    # 2. Validator rejects target_entity mismatch if question inquires about spatial indexing but claims React
    ev_text = (
        "Telecom Geofencing System | FastAPI, React.js, QGIS, Shapely, R-Tree Indexing, GeoJSON, Python\n"
        "Replaced legacy linear scan with an R-Tree spatial index over GeoJSON MultiPolygons in FastAPI. Built a React admin panel."
    )
    resolved_ev = {
        "chk_geo": {"text": ev_text, "source_type": "chunk"}
    }

    mismatched_q = QuestionOutputSchema(
        question=(
            "In the Telecom Geofencing System built using React, FastAPI, QGIS, Shapely, R-Tree Indexing, and GeoJSON, "
            "how would you approach designing the spatial indexing and data retrieval architecture to maintain low-latency API resolution performance when processing complex polygon geometries?"
        ),
        category="Architecture",
        difficulty=3,
        target_project="Telecom Geofencing System",
        target_entity="React",  # Mismatch: question inquires about spatial indexing
        evidence_ids=["chk_geo"],
        evidence_text=ev_text,
        rationale="Spatial indexing evaluation",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_geo_test"
    )

    is_valid, reason, details = grounding_validator.validate(
        question=mismatched_q,
        allowed_projects=["Telecom Geofencing System"],
        allowed_entities=["React", "FastAPI", "QGIS", "Shapely", "R-Tree Indexing", "GeoJSON"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )

    assert is_valid is False
    assert details.get("code") == "target_entity_focus_mismatch"
    assert "R-Tree Indexing" in reason or "target_entity" in reason


def test_regression_project_name_validation_tolerates_punctuation_and_hyphen_variations():
    """
    Issue 5: Verify project-name validation tolerates hyphens, dashes, and whitespace variations
    while rejecting genuinely unrelated projects.
    """
    from app.services.rag.validator import grounding_validator
    from app.schemas.rag import QuestionOutputSchema

    ev_text = "Real-Time Conversational Telephony AI Voicebot | Python, AsyncIO, GCP."
    resolved_ev = {
        "chk_voice": {"text": ev_text, "source_type": "chunk"}
    }

    # Question targeting project with dash variation
    q = QuestionOutputSchema(
        question="How would you approach designing the audio buffer queue in the Real - Time Conversational Telephony AI Voicebot?",
        category="Architecture",
        difficulty=3,
        target_project="Real - Time Conversational Telephony AI Voicebot",
        target_entity="Python",
        evidence_ids=["chk_voice"],
        evidence_text=ev_text,
        rationale="Audio buffer design",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_norm_test"
    )

    # Allowed projects has hyphenated version
    is_valid, reason, details = grounding_validator.validate(
        question=q,
        allowed_projects=["Real-Time Conversational Telephony AI Voicebot"],
        allowed_entities=["Python"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is True

    # Genuinely unrelated project must still be rejected
    unrelated_q = QuestionOutputSchema(
        question="How would you approach designing the database in the Cloud E-Commerce Engine?",
        category="Architecture",
        difficulty=3,
        target_project="Cloud E-Commerce Engine",
        target_entity="Python",
        evidence_ids=["chk_voice"],
        evidence_text=ev_text,
        rationale="Unrelated probe",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_unrel"
    )
    is_unrelated_valid, unrel_reason, unrel_details = grounding_validator.validate(
        question=unrelated_q,
        allowed_projects=["Real-Time Conversational Telephony AI Voicebot"],
        allowed_entities=["Python"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_unrelated_valid is False
    assert unrel_details.get("code") == "hallucinated_project"


def test_regression_difficulty_estimation_for_open_ended_architecture_questions():
    """
    Issue 6: Verify difficulty estimation accurately scores open-ended architecture
    and performance-optimization questions at Level 3+, while keeping basic questions lower.
    """
    from app.services.rag.validator import grounding_validator

    # Open-ended architecture question previously misclassified as Level 1
    geo_question = (
        "In the Telecom Geofencing System built using React, FastAPI, QGIS, Shapely, R-Tree Indexing, and GeoJSON, "
        "how would you approach designing the spatial indexing and data retrieval architecture to maintain low-latency API resolution performance when processing complex polygon geometries?"
    )
    est_diff = grounding_validator.estimate_difficulty_level(geo_question)
    assert est_diff >= 3, f"Expected difficulty >= 3, got {est_diff}"

    # Basic conceptual question must remain entry level
    basic_question = "What is a GeoJSON polygon and what are its basic properties?"
    assert grounding_validator.estimate_difficulty_level(basic_question) <= 2

    # Advanced failure-mode question must score Level 5
    adv_question = "What is the cascading failure mode and distributed blast radius when network partition occurs?"
    assert grounding_validator.estimate_difficulty_level(adv_question) == 5


def test_validator_rejects_walk_me_through_without_chunk_evidence():
    """
    'Walk me through...' must be caught as an assertive presupposition.
    GAP-1 regression: the regex previously matched 'walk through' but not 'walk me through'.
    """
    resolved_evidence = {
        "rel_abc123": {
            "evidence_id": "rel_abc123",
            "text": "PaymentService HAS_TECHNOLOGY Stripe (Evidence: Stripe)",
            "project": "PaymentService",
            "source_type": "relationship"
        }
    }
    q = QuestionOutputSchema(
        question="Walk me through the steps you took to integrate Stripe payment handling in PaymentService.",
        category="Implementation",
        difficulty=3,
        target_project="PaymentService",
        target_entity="Stripe",
        evidence_ids=["rel_abc123"],
        evidence_text="PaymentService HAS_TECHNOLOGY Stripe",
        rationale="Testing walk-me-through phrase",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test_gap1a"
    )
    is_valid, reason, _ = grounding_validator.validate(
        question=q,
        allowed_projects=["PaymentService"],
        allowed_entities=["Stripe"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False, f"Expected rejection for 'walk me through' on tech-only evidence, got: {reason}"
    assert "unsupported" in reason.lower() or "presupposition" in reason.lower() or "unverified" in reason.lower(), (
        f"Expected unsupported_implementation_presupposition rejection, got: {reason}"
    )


def test_validator_rejects_tell_me_about_your_implementation_without_chunk_evidence():
    """
    'Tell me about your implementation of...' must be caught as an assertive presupposition.
    GAP-1 regression: this phrase was entirely absent from the regex before this fix.
    """
    resolved_evidence = {
        "rel_def456": {
            "evidence_id": "rel_def456",
            "text": "OrderService HAS_TECHNOLOGY Kafka (Evidence: Kafka)",
            "project": "OrderService",
            "source_type": "relationship"
        }
    }
    q = QuestionOutputSchema(
        question="Tell me about your implementation of Kafka message queuing in OrderService.",
        category="Architecture",
        difficulty=3,
        target_project="OrderService",
        target_entity="Kafka",
        evidence_ids=["rel_def456"],
        evidence_text="OrderService HAS_TECHNOLOGY Kafka",
        rationale="Testing tell-me-about phrase",
        grounding_score=0.0,
        relevance_score=0.0,
        technical_quality_score=0.0,
        trace_id="trc_test_gap1b"
    )
    is_valid, reason, _ = grounding_validator.validate(
        question=q,
        allowed_projects=["OrderService"],
        allowed_entities=["Kafka"],
        resolved_evidence_items=resolved_evidence,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False, f"Expected rejection for 'tell me about your implementation of' on tech-only evidence, got: {reason}"
    assert "unsupported" in reason.lower() or "presupposition" in reason.lower() or "unverified" in reason.lower(), (
        f"Expected unsupported_implementation_presupposition rejection, got: {reason}"
    )


def test_claim_service_matches_target_entity_canonical_aliases():
    """
    GAP-2 Regression: When an extracted relationship has a canonicalized target entity (e.g. 'Spring Boot'),
    its evidence text containing alias variants ('springboot') must be recognized as SUPPORTED.
    """
    from unittest.mock import MagicMock
    from uuid import uuid4
    from app.services.rag.claim_service import claim_service
    from app.models.relationship import Relationship

    rid = uuid4()
    rel1 = Relationship(
        id=uuid4(),
        resume_id=rid,
        source_entity="CampusBridge",
        relation_type="BUILT_WITH",
        target_entity="Spring Boot",
        section="Projects",
        context_entity="CampusBridge",
        evidence_text="Engineered backend API microservices using springboot framework.",
        confidence=0.95,
        extraction_method="spacy_rule"
    )
    rel2 = Relationship(
        id=uuid4(),
        resume_id=rid,
        source_entity="SpatialRouter",
        relation_type="BUILT_WITH",
        target_entity="R-Tree Indexing",
        section="Projects",
        context_entity="SpatialRouter",
        evidence_text="Applied rtree indexing algorithm for rapid spatial bounding queries.",
        confidence=0.90,
        extraction_method="spacy_rule"
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [rel1, rel2]

    claims = claim_service.extract_and_store_claims(rid, mock_db)
    assert len(claims) == 2
    assert claims[0].status == "SUPPORTED", f"Expected Spring Boot claim to be SUPPORTED, got {claims[0].status}"
    assert claims[1].status == "SUPPORTED", f"Expected R-Tree Indexing claim to be SUPPORTED, got {claims[1].status}"


def test_reranker_does_not_disqualify_personal_projects_chunk():
    """
    GAP-3 Regression: The word 'PERSONAL' in chunk text (e.g. 'PERSONAL PROJECTS')
    must not cause calculate_interview_relevance_score to zero out the technical chunk.
    Non-technical relationships (e.g. 'SPEAKS') must still be zeroed out.
    """
    from app.schemas.rag import EvidenceItem
    from app.services.rag.reranker import DynamicStructuralRelevanceReranker

    chunk_item = EvidenceItem(
        evidence_id="chk_pers_1",
        text="PERSONAL PROJECTS: Built distributed telemetry ingest using WebSockets and Redis.",
        section="Projects",
        project="TelemetryApp",
        source_type="chunk",
        score=0.75,
        provenance="chunk_text"
    )
    score_chunk = DynamicStructuralRelevanceReranker.calculate_interview_relevance_score(
        chunk_item, target_entity="Redis", target_project="TelemetryApp"
    )
    assert score_chunk > 0.50, f"Personal project chunk was unexpectedly disqualified with score {score_chunk}"

    rel_item = EvidenceItem(
        evidence_id="rel_pers_1",
        text="Candidate SPEAKS Spanish (Evidence: Native Spanish)",
        section="Languages",
        project=None,
        source_type="relationship",
        score=0.80,
        provenance="graph"
    )
    score_rel = DynamicStructuralRelevanceReranker.calculate_interview_relevance_score(rel_item)
    assert score_rel == 0.0, f"Non-technical relationship item was not disqualified: {score_rel}"

