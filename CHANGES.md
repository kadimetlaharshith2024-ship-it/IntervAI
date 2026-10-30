# IntervAI Production-Hardening Ledger of Changes

This document records the architectural hardening modifications implemented across IntervAI's backend and data layers across Tasks 1 through 5.

---

## 1. Summary of Changes by Task

### Task 1: PII Sanitization & Redaction
- **`backend/app/services/privacy_guard.py`**:
  - Implemented multi-pattern regex redaction for labelled contact lines (`T:`, `Tel:`, `Ph:`, `Mob:`, `Email:`, `Contact:`), 10-digit/12-digit Indian mobile formats (`+91`), US telephone formats, email variants (including obfuscated formats like `[at]`, `(dot)`), and social handles/profile URLs (LinkedIn, GitHub).
  - Defined `ALL_REDACTION_MARKERS` (`[REDACTED_PHONE]`, `[REDACTED_EMAIL]`, `[REDACTED_LINKEDIN]`, `[REDACTED_GITHUB]`, `[REDACTED_NAME]`).
- **`backend/app/services/resume_parser.py`**:
  - Bound privacy sanitization directly at document ingestion (`privacy_guard.sanitize_text`), ensuring raw PII is never passed downstream to entity extractors, relationship extractors, or chunking engines.
- **`backend/app/services/rag/chunking_service.py`**:
  - Hardened chunk filtering to explicitly drop blocks classified as `header` or `contact`, guaranteeing contact details never enter semantic vector embeddings.
- **`backend/alembic/versions/a1b2c3d4e5f6_resanitize_pii_in_stored_records.py`**:
  - Applied retroactive data migration `a1b2c3d4e5f6` executing idempotent sanitization over existing `Resume`, `ResumeChunk`, `ResumeClaim`, `Relationship`, and `InterviewQuestion` records.
- **`scripts/resanitize_database.py`**:
  - Standalone operational script for maintenance-mode PII scrub and session-state sanitization.

---

### Task 2: Clean Topic Extraction & Assessable Taxonomy
- **`backend/app/services/knowledge_state_service.py`**:
  - Created `JUNK_TOPIC_BLOCKLIST` blocking generic non-assessable terms (`candidate`, `institute`, `college`, `university`, `degree`, `b.tech`, `m.tech`, `semester`, `cgpa`, `rank`, `achievements`, `languages`, `extracurricular`, sports, fests).
  - Implemented `is_assessable_topic()` predicate: topics must strictly originate from either canonical technical taxonomy or verified project/experience names (`is_assessable=False` otherwise).
  - Re-anchored topic coverage calculation (`coverage_percentage`) to compute exclusively over assessable topics.
- **`backend/app/services/adaptive_selection_service.py`**:
  - Filtered adaptive candidate generation to enforce `is_assessable=True` for uncovered topics, coverage expansion, and gap follow-ups.

---

### Task 3: Evidence Gate & Entity Grounding
- **`backend/app/services/rag/retrievers.py`**:
  - Enforced strict evidence threshold requiring $\ge 1$ verified evidence item (chunk, relationship, or claim) from projects or experience sections for any project-backed question.
  - Skills-list-only topics sort last, strictly restricted to `Fundamentals` conceptual categories, and are prohibited from hallucinating project contexts.
- **`backend/app/services/rag/rag_pipeline.py`**:
  - Rejection gate: if `target_entity` is absent from cited evidence, questions are deterministically rejected with `EVIDENCE_UNAVAILABLE` (error code: `evidence_missing_target`).
  - Added duplicate check mechanisms across LLM structured generators to eliminate lexical and semantic collision across turns.

---

### Task 4: Knowledge Gap & Adaptive Selection Logic
- **`backend/app/services/knowledge_state_service.py`**:
  - Calibrated gap registration criteria: knowledge gaps are only logged if score $< 75$ or ($< 85$ with substantive `missing_concepts`).
  - Mastery threshold $\ge 85$ strictly prohibits subsequent gap follow-ups on the mastered topic.
  - Refined claim verification tracking: general questions on the same project/technology no longer prematurely flip `ResumeClaim` status to `SUPPORTED`. Only questions targeting the claim update its verification status.
- **`backend/app/services/adaptive_selection_service.py`**:
  - Capped topic follow-ups to a maximum of 2 consecutive questions.
  - Enforced mandatory breadth question rotation after any follow-up sequence prior to revisiting knowledge gaps.
  - Guarded claim and project depth quotas ahead of difficulty escalation to avoid quota starvation.

---

### Task 5: Latency Budget & Model Warmup
- **`backend/app/core/config.py`**:
  - Added `TOTAL_QUESTION_DEADLINE_SECONDS = 12.0`: global ceiling for question generation turn.
  - Added `MIN_LLM_CALL_BUDGET = 2.5`: minimum remaining time required to trigger a second LLM generation attempt.
  - Added `MAX_CANDIDATE_ATTEMPTS = 2`: maximum rotation attempts across candidate targets.
  - Added `MAX_INTERNAL_RAG_ATTEMPTS = 2`: maximum internal RAG generation attempts per turn.
- **`backend/app/services/rag/rag_pipeline.py`**:
  - Dynamic deadline budget propagation: sets LLM call timeout to $\min(\text{configured\_timeout}, \text{remaining\_budget})$.
  - Aborts Attempt 2 when remaining budget is less than `MIN_LLM_CALL_BUDGET` (2.5s), returning a recoverable message (`"Question is still preparing, please retry"`) with HTTP 200/recoverable status without terminating or corrupting the session.
  - Instrumented `retrieval_traces.latencies_ms` with granular breakdowns:
    - `db_preload`
    - `retrieval`
    - `rerank`
    - `context_assembly`
    - `llm_call_1`
    - `llm_call_2`
    - `validation`
    - `total_ms`
- **`backend/app/main.py`**:
  - Added startup warmup handlers for FastEmbed embeddings model (`embedding_service.embed_texts(["warmup"])`) and spaCy NLP pipeline; eliminated runtime cold-start model downloads.
- **`backend/app/services/session_service.py`**:
  - Pre-generates Question 1 asynchronously during `create_session`, pre-caching Turn 1 for sub-100ms first question delivery.

---

## 2. Configuration Parameters Added

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `TOTAL_QUESTION_DEADLINE_SECONDS` | `float` | `12.0` | Global deadline for generating an adaptive question |
| `MIN_LLM_CALL_BUDGET` | `float` | `2.5` | Minimum seconds remaining to allow Attempt 2 LLM generation |
| `MAX_CANDIDATE_ATTEMPTS` | `int` | `2` | Maximum candidate target rotations |
| `MAX_INTERNAL_RAG_ATTEMPTS` | `int` | `2` | Maximum generation retries per target |

---

## 3. Database Migrations Applied

- **`a1b2c3d4e5f6_resanitize_pii_in_stored_records`**:
  - Applies retroactive regex redaction across raw text, chunks, claims, relationships, and questions.
  - Cleans legacy `InterviewSession.session_state` records by dropping non-assessable junk topics and recomputing coverage metrics over assessable topics only.

---

## 4. Verification Test Matrix

All 447 tests across all 41 test files in `backend/tests/` passed with 0 failures:
- `tests/test_hardening_task1_pii.py` (PII redaction, contact lines, Indian mobiles) -> **PASSED**
- `tests/test_hardening_task2_topics.py` (Junk topic blocklist, assessable flag, coverage) -> **PASSED**
- `tests/test_hardening_task3_evidence_gate.py` (Evidence gate, target entity presence, skills-only) -> **PASSED**
- `tests/test_hardening_task4_gap_logic.py` (Gap thresholds, mastery $\ge 85$, follow-up caps, breadth rotation) -> **PASSED**
- `tests/test_hardening_task5_latency.py` (12.0s deadline, attempt 2 abort, startup warmup, trace latencies) -> **PASSED**
- `tests/test_simulation_10q.py` (10-question simulations: strong, weak, refusing, prefetch) -> **PASSED**
- `tests/test_phase5_rag.py` (49 comprehensive RAG pipeline tests) -> **PASSED**
- `tests/test_privacy_reprocessing_lifecycle.py` -> **PASSED**
- `tests/test_phase7_*.py` (Knowledge state, adaptive selection, safety recovery) -> **PASSED**
