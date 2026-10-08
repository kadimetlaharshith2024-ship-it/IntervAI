# IntervAI — Development Progress

Here’s the development roadmap for IntervAI.

---

# Phase 1: Project Setup & Database

- [ ] Create FastAPI Project
- [ ] Configure Python Virtual Environment
- [ ] Configure Project Structure
- [ ] Configure PostgreSQL
- [ ] Configure SQLAlchemy
- [ ] Configure `.env`
- [ ] Configure Database Connection
- [ ] Create Base Model
- [ ] Create Database Session
- [ ] Create Alembic Configuration
- [ ] Create Resume Model
- [ ] Create Skill Model
- [ ] Create ResumeSkill Model
- [ ] Create Relationship Model
- [ ] Create InterviewSession Model
- [ ] Create InterviewQuestion Model
- [ ] Create InterviewAnswer Model
- [ ] Add Primary Keys
- [ ] Add Foreign Keys
- [ ] Add Unique Constraints
- [ ] Add Required Indexes
- [ ] Create Initial Migration
- [ ] Run Database Migration
- [ ] Test PostgreSQL Connection

### Milestone:

- FastAPI project runs successfully
- PostgreSQL connection works
- All base tables are created successfully

---

# Phase 2: Resume Ingestion Module

- [ ] Create Resume Upload Route
- [ ] Create Resume Upload Service
- [ ] Implement PDF Validation
- [ ] Implement File Size Validation
- [ ] Integrate `pdfplumber`
- [ ] Extract Text from PDF
- [ ] Clean Extracted Text
- [ ] Normalize Whitespace
- [ ] Remove Encoding Artifacts
- [ ] Detect Resume Sections
- [ ] Detect Skills Section
- [ ] Detect Projects Section
- [ ] Detect Experience Section
- [ ] Detect Education Section
- [ ] Save Resume Metadata
- [ ] Save Raw Resume Text
- [ ] Add Processing Status
- [ ] Test Upload API in Postman
- [ ] Test Multi-page Resume
- [ ] Test Invalid PDF
- [ ] Test Non-PDF File

### Milestone:

- Resume PDF can be uploaded
- Text is extracted correctly
- Resume information is stored in PostgreSQL

---

# Phase 3: Entity Extraction Module

- [ ] Integrate Hugging Face Transformers
- [ ] Configure Transformer Model
- [ ] Create Entity Extraction Service
- [ ] Extract Programming Languages
- [ ] Extract Frameworks
- [ ] Extract Libraries
- [ ] Extract Databases
- [ ] Extract Cloud Technologies
- [ ] Extract Infrastructure Tools
- [ ] Extract Project Names
- [ ] Extract Quantifiable Metrics
- [ ] Normalize Extracted Entities
- [ ] Remove Duplicate Entities
- [ ] Add Entity Confidence Score
- [ ] Store Skills in `skills`
- [ ] Store Resume-Skill Mapping in `resume_skills`
- [ ] Create Entity Extraction API
- [ ] Test Entity Extraction with Resume
- [ ] Verify Extracted Entities Against Resume Text

### Milestone:

- Resume is converted from raw text into structured technical entities
- Extracted skills are stored in PostgreSQL

---

# Phase 4: Relationship Extraction Module

- [ ] Define Relationship Types
  - [ ] USED_FOR
  - [ ] IMPLEMENTED_WITH
  - [ ] INTEGRATED_WITH
  - [ ] DEPLOYED_ON
  - [ ] OPTIMIZED_VIA
  - [ ] BUILT_USING

- [ ] Create Relationship Extraction Service
- [ ] Detect Entity-to-Entity Relationships
- [ ] Generate Relationship Triplets
- [ ] Generate Relationship Confidence Score
- [ ] Store Relationships in PostgreSQL
- [ ] Create Relationship Query Service
- [ ] Create Relationship API
- [ ] Implement `GET /api/resumes/{id}/relationships`
- [ ] Test Simple Relationships
- [ ] Test Multiple Relationships
- [ ] Test Project-Based Relationships
- [ ] Verify Relationship Triplets Against Resume

### Example:

(JWT, USED_FOR, Authentication)

(Spring Security, IMPLEMENTS, JWT)

(Redis, USED_FOR, Caching)

(Spring Boot, INTEGRATED_WITH, PostgreSQL)

### Milestone:

- System understands relationships between technologies, projects, and implementation claims

---

# Phase 5: Production RAG & Question Generation Module

## 5.1 RAG Architecture

- [ ] Configure RAG Pipeline
- [ ] Define RAG Architecture
- [ ] Define Retrieval Interfaces
- [ ] Define Context Object Schema
- [ ] Define Retrieval Metadata Schema
- [ ] Define Retrieval Trace Schema
- [ ] Define RAG Failure Handling
- [ ] Define Retrieval Configuration
- [ ] Add RAG Pipeline Logging
- [ ] Add RAG Request/Trace ID

## 5.2 Embedding Layer

- [ ] Select Embedding Model
- [ ] Evaluate Embedding Model for Resume Text
- [ ] Configure Embedding Dimensions
- [ ] Configure Embedding Generation Service
- [ ] Implement Embedding Batching
- [ ] Implement Embedding Error Handling
- [ ] Implement Embedding Retry Logic
- [ ] Track Embedding Latency
- [ ] Track Embedding Usage/Cost
- [ ] Version Embedding Model
- [ ] Store Embedding Model Metadata

## 5.3 Resume Chunking

- [ ] Create Resume Chunking Strategy
- [ ] Implement Section-Aware Chunking
- [ ] Implement Semantic Chunking
- [ ] Preserve Project Boundaries
- [ ] Preserve Experience Boundaries
- [ ] Preserve Education Boundaries
- [ ] Preserve Certification Boundaries
- [ ] Preserve Evidence Text
- [ ] Preserve Source Section
- [ ] Preserve Source Location/Page Metadata
- [ ] Generate Stable Chunk IDs
- [ ] Store Chunk Metadata
- [ ] Prevent Meaningless Chunk Splits
- [ ] Prevent Excessively Large Chunks
- [ ] Test Chunking on Multi-Page Resume
- [ ] Test Chunking on Different Resume Structures

## 5.4 Vector Storage

- [ ] Configure Vector Storage
- [ ] Select Vector Index
- [ ] Create Vector Document Schema
- [ ] Store Resume Embeddings
- [ ] Store Chunk Metadata
- [ ] Store Resume ID with Embedding
- [ ] Store Section Metadata
- [ ] Store Project Metadata
- [ ] Store Entity Metadata
- [ ] Store Relationship Metadata
- [ ] Implement Vector Upsert
- [ ] Implement Vector Delete
- [ ] Implement Vector Update
- [ ] Prevent Duplicate Embeddings
- [ ] Version Indexed Documents

## 5.5 Knowledge Graph Retrieval

- [ ] Create Graph Context Retriever
- [ ] Retrieve Related Entities
- [ ] Retrieve Relationship Triplets
- [ ] Retrieve Project Context
- [ ] Retrieve Candidate-Level Facts
- [ ] Retrieve Relationship Evidence
- [ ] Retrieve Relationship Confidence
- [ ] Retrieve Relationship Provenance
- [ ] Filter Relationships by Resume ID
- [ ] Filter Relationships by Project
- [ ] Filter Relationships by Section
- [ ] Prevent Cross-Resume Retrieval

## 5.6 Hybrid Retrieval

- [ ] Implement Vector Retrieval
- [ ] Implement Graph Retrieval
- [ ] Implement Keyword/Lexical Retrieval
- [ ] Implement Metadata Filtering
- [ ] Implement Hybrid Result Fusion
- [ ] Configure Top-K Retrieval
- [ ] Implement Retrieval Score Normalization
- [ ] Implement Result Deduplication
- [ ] Implement Context Relevance Filtering
- [ ] Implement Reranking
- [ ] Configure Reranker
- [ ] Evaluate Reranker
- [ ] Preserve Evidence During Reranking
- [ ] Implement Context Token Budget
- [ ] Prevent Irrelevant Context Injection
- [ ] Implement Retrieval Fallback

## 5.7 Claim Intelligence

- [ ] Define Resume Claim Schema
- [ ] Extract Candidate Claims
- [ ] Link Claims to Evidence
- [ ] Link Claims to Entities
- [ ] Link Claims to Relationships
- [ ] Link Claims to Projects
- [ ] Assign Claim Confidence
- [ ] Track Claim Provenance
- [ ] Define Claim Status
  - [ ] UNTESTED
  - [ ] PARTIALLY_TESTED
  - [ ] SUPPORTED
  - [ ] WEAKLY_SUPPORTED
  - [ ] CONTRADICTED
  - [ ] VERIFIED
- [ ] Prevent Unsupported Claims from Entering Question Context
- [ ] Preserve Claim Evidence

## 5.8 Context Assembly

- [ ] Create Resume Context Retriever
- [ ] Retrieve Relevant Resume Sections
- [ ] Retrieve Related Entities
- [ ] Retrieve Relationship Triplets
- [ ] Retrieve Resume Claims
- [ ] Combine Retrieved Context
- [ ] Rank Evidence by Relevance
- [ ] Prioritize Direct Evidence
- [ ] Remove Duplicate Evidence
- [ ] Remove Conflicting Low-Confidence Context
- [ ] Build Structured LLM Context
- [ ] Enforce Context Token Budget
- [ ] Store Retrieval Trace

## 5.9 Question Generation

- [ ] Create Question Generation Service
- [ ] Configure LLM
- [ ] Version LLM Configuration
- [ ] Create Structured Question Output Schema
- [ ] Create Question Generation Prompt
- [ ] Create Fundamentals Question Logic
- [ ] Create Application Question Logic
- [ ] Create Implementation Question Logic
- [ ] Create Architecture Question Logic
- [ ] Create Performance Question Logic
- [ ] Create Security Question Logic
- [ ] Create Trade-off Question Logic
- [ ] Create Failure-Mode Question Logic
- [ ] Create Claim-Verification Question Logic
- [ ] Create Relationship-Based Question Logic
- [ ] Create Project-Based Question Logic
- [ ] Create Cross-Section Question Logic

## 5.10 Difficulty

- [ ] Implement Difficulty Levels
  - [ ] Level 1 — Fundamentals
  - [ ] Level 2 — Application
  - [ ] Level 3 — Implementation
  - [ ] Level 4 — Architecture
  - [ ] Level 5 — Advanced Depth
- [ ] Define Difficulty Criteria
- [ ] Validate Difficulty Against Question Content
- [ ] Prevent Difficulty Mismatch
- [ ] Store Difficulty Metadata

## 5.11 Grounding & Quality Control

- [ ] Implement Grounding Validation
- [ ] Validate Question Against Retrieved Evidence
- [ ] Validate Question Against Resume Claims
- [ ] Validate Question Against Relationships
- [ ] Validate Question Against Project Context
- [ ] Reject Unsupported Resume Claims
- [ ] Reject Unsupported Technologies
- [ ] Reject Unsupported Architecture Claims
- [ ] Reject Unsupported Deployment Claims
- [ ] Implement Question Evidence Requirement
- [ ] Implement Question Relevance Score
- [ ] Implement Question Technical Quality Score
- [ ] Implement Structured Output Validation
- [ ] Implement LLM Output Retry
- [ ] Implement LLM Output Fallback
- [ ] Implement Hallucination Detection
- [ ] Implement Duplicate Question Detection
- [ ] Implement Near-Duplicate Question Detection
- [ ] Prevent Repeated Question Intent
- [ ] Prevent Generic Questions When Strong Resume Evidence Exists

## 5.12 Persistence

- [ ] Link Question to Resume
- [ ] Link Question to Resume Claim
- [ ] Link Question to Relationship
- [ ] Link Question to Project
- [ ] Store Question Evidence
- [ ] Store Question Difficulty
- [ ] Store Question Category
- [ ] Store Retrieval Metadata
- [ ] Store Generation Metadata
- [ ] Store Question Validation Result
- [ ] Store Questions in PostgreSQL

## 5.13 API & Testing

- [ ] Create Question Generation API
- [ ] Validate API Input
- [ ] Validate API Output
- [ ] Implement API Error Handling
- [ ] Test Question Generation in Postman
- [ ] Test Questions from Multiple Resume Sections
- [ ] Test Project-Specific Questions
- [ ] Test Relationship-Based Questions
- [ ] Test Claim-Verification Questions
- [ ] Test Unsupported Claim Rejection
- [ ] Test Duplicate Detection
- [ ] Test Retrieval Failure
- [ ] Test LLM Failure
- [ ] Test Empty Retrieval Result
- [ ] Test Large Context
- [ ] Test Multi-Project Resume
- [ ] Verify PostgreSQL Data
- [ ] Verify Retrieval Trace
- [ ] Verify Evidence Provenance

### Phase 5 Milestone

- [ ] Hybrid vector + graph retrieval works
- [ ] Resume evidence is retrieved correctly
- [ ] Candidate claims are grounded
- [ ] Questions are generated from actual resume evidence
- [ ] Unsupported questions are rejected
- [ ] Duplicate questions are prevented
- [ ] Question provenance is stored
- [ ] Retrieval and generation are observable


# Phase 6: Production Answer Evaluation Module

## 6.1 Answer Processing

- [ ] Create Answer Submission API
- [ ] Validate Answer Input
- [ ] Normalize Answer Text
- [ ] Handle Empty Answer
- [ ] Handle Very Short Answer
- [ ] Handle Very Long Answer
- [ ] Create Answer Evaluation Service
- [ ] Link Answer to Interview Question
- [ ] Link Answer to Interview Session
- [ ] Store Candidate Answer

## 6.2 Evaluation Rubric

- [ ] Define Answer Evaluation Rubric
- [ ] Define Scoring Scale
- [ ] Evaluate Technical Accuracy
- [ ] Evaluate Conceptual Understanding
- [ ] Evaluate Completeness
- [ ] Evaluate Depth of Explanation
- [ ] Evaluate Implementation Knowledge
- [ ] Evaluate Architectural Reasoning
- [ ] Evaluate Problem-Solving Reasoning
- [ ] Evaluate Trade-off Understanding
- [ ] Evaluate Security Understanding
- [ ] Evaluate Performance Understanding
- [ ] Evaluate Failure-Mode Understanding
- [ ] Evaluate Evidence Consistency

## 6.3 Knowledge Gap Detection

- [ ] Detect Missing Concepts
- [ ] Detect Incorrect Concepts
- [ ] Detect Partially Correct Concepts
- [ ] Detect Knowledge Gaps
- [ ] Detect Shallow Understanding
- [ ] Detect Unsupported Candidate Claims
- [ ] Detect Contradictions
- [ ] Identify Weak Technologies
- [ ] Identify Weak Relationships
- [ ] Identify Weak Projects
- [ ] Identify Missing Reasoning

## 6.4 Evaluation Engine

- [ ] Configure Evaluation Prompt
- [ ] Create Structured Evaluation Schema
- [ ] Implement LLM Output Validation
- [ ] Implement Evaluation Confidence
- [ ] Implement Evaluation Evidence
- [ ] Implement Evaluation Retry
- [ ] Implement Evaluation Fallback
- [ ] Prevent Unsupported Evaluation Claims
- [ ] Validate Evaluation Against Question
- [ ] Validate Evaluation Against Expected Concepts
- [ ] Compare Answer Against Resume Context

## 6.5 Scoring

- [ ] Generate Answer Score
- [ ] Generate Category Scores
- [ ] Generate Strengths
- [ ] Generate Weaknesses
- [ ] Generate Missing Concepts
- [ ] Generate Knowledge Gaps
- [ ] Generate Improvement Suggestions
- [ ] Generate Evaluation Feedback
- [ ] Calculate Session Score
- [ ] Track Score History

## 6.6 Persistence

- [ ] Store Score
- [ ] Store Feedback
- [ ] Store Knowledge Gaps
- [ ] Store Missing Concepts
- [ ] Store Evaluation Metadata
- [ ] Store Evaluation Confidence
- [ ] Store Evaluation Evidence
- [ ] Update Interview Session Score

## 6.7 Testing

- [ ] Test Strong Answer
- [ ] Test Medium Answer
- [ ] Test Weak Answer
- [ ] Test Partially Correct Answer
- [ ] Test Irrelevant Answer
- [ ] Test Hallucinated Answer
- [ ] Test Contradictory Answer
- [ ] Test Very Long Answer
- [ ] Test Empty Answer
- [ ] Test Multiple Question Categories

### Phase 6 Milestone

- [ ] Candidate answers are reliably evaluated
- [ ] Scores are explainable
- [ ] Knowledge gaps are identified
- [ ] Evaluation is grounded in the question and resume context
- [ ] Evaluation results are persisted


# Phase 7: Production Adaptive Interview Module

## 7.1 Interview Session

- [ ] Create Interview Session Service
- [ ] Create Start Interview API
- [ ] Create End Interview API
- [ ] Create Interview State Management
- [ ] Generate Session ID
- [ ] Track Session Status
- [ ] Track Interview Start Time
- [ ] Track Interview End Time
- [ ] Track Asked Questions
- [ ] Track Answer History
- [ ] Track Tested Skills
- [ ] Track Tested Relationships
- [ ] Track Tested Claims
- [ ] Track Knowledge Gaps
- [ ] Track Current Difficulty
- [ ] Track Interview Progress

## 7.2 Candidate Knowledge State

- [ ] Create Candidate Knowledge State
- [ ] Track Skill Mastery
- [ ] Track Technology Mastery
- [ ] Track Project Understanding
- [ ] Track Relationship Understanding
- [ ] Track Claim Verification State
- [ ] Track Knowledge Gaps
- [ ] Track Strong Areas
- [ ] Track Weak Areas
- [ ] Update Knowledge State After Every Answer

## 7.3 Adaptive Question Engine

- [ ] Implement Adaptive Question Logic
- [ ] Implement Next-Best-Question Logic
- [ ] Score Question Candidates
- [ ] Consider Knowledge Gaps
- [ ] Consider Previous Answers
- [ ] Consider Difficulty
- [ ] Consider Topic Coverage
- [ ] Consider Relationship Coverage
- [ ] Consider Claim Coverage
- [ ] Consider Question History
- [ ] Consider Information Gain
- [ ] Prevent Redundant Questions

## 7.4 Score-Based Adaptation

- [ ] Score < 5
  - [ ] Identify Weak Concept
  - [ ] Generate Targeted Follow-up
  - [ ] Reduce Difficulty When Appropriate

- [ ] Score 5–7.5
  - [ ] Ask Implementation Question
  - [ ] Ask Trade-off Question
  - [ ] Ask Reasoning Question

- [ ] Score > 7.5
  - [ ] Increase Difficulty
  - [ ] Ask Architecture Question
  - [ ] Ask Failure-Mode Question
  - [ ] Ask Advanced Depth Question

## 7.5 Follow-Up Intelligence

- [ ] Implement Parent Question ID
- [ ] Implement Child Question Relationship
- [ ] Store Follow-Up Reason
- [ ] Prevent Repeated Questions
- [ ] Prevent Repeated Topics
- [ ] Prevent Repeated Relationships
- [ ] Prevent Repeated Claims
- [ ] Implement Difficulty Escalation
- [ ] Implement Weakness Probing
- [ ] Implement Strong-Answer Escalation
- [ ] Implement Partial-Answer Follow-up
- [ ] Implement Resume-Claim Verification Questions
- [ ] Implement Contradiction Follow-up
- [ ] Implement Clarification Follow-up

## 7.6 Interview Safety & Reliability

- [ ] Validate Generated Follow-Up
- [ ] Ground Follow-Up in Resume Context
- [ ] Prevent Unsupported Topics
- [ ] Prevent Infinite Follow-Up Loops
- [ ] Define Maximum Interview Length
- [ ] Define Maximum Questions per Topic
- [ ] Implement Session Recovery
- [ ] Implement Failure Recovery

## 7.7 Testing

- [ ] Test 5-Turn Interview Flow
- [ ] Test Weak-Answer Flow
- [ ] Test Strong-Answer Flow
- [ ] Test Mixed-Answer Flow
- [ ] Test Partial-Answer Flow
- [ ] Test Contradiction Flow
- [ ] Test Claim-Verification Flow
- [ ] Test Difficulty Escalation
- [ ] Test Duplicate Prevention
- [ ] Test Session Recovery

### Phase 7 Milestone

- [ ] Interview dynamically changes based on candidate performance
- [ ] Strong answers increase difficulty
- [ ] Weak answers trigger targeted follow-ups
- [ ] Partial answers trigger deeper probing
- [ ] Questions are selected using candidate knowledge state
- [ ] Repetition is prevented


# Phase 8: Production Candidate Analytics, Final Report & Optional Voice

## 8.1 Analytics Engine

- [ ] Create Interview Analytics Service
- [ ] Calculate Overall Interview Score
- [ ] Calculate Category Scores
- [ ] Calculate Backend Performance
- [ ] Calculate Frontend Performance
- [ ] Calculate Database Performance
- [ ] Calculate Security Performance
- [ ] Calculate System Design Performance
- [ ] Calculate Technical Depth
- [ ] Calculate Problem-Solving Performance
- [ ] Calculate Architecture Performance
- [ ] Calculate Performance Reasoning
- [ ] Calculate Consistency

## 8.2 Candidate Intelligence

- [ ] Identify Strengths
- [ ] Identify Weaknesses
- [ ] Identify Knowledge Gaps
- [ ] Track Tested Skills
- [ ] Track Tested Relationships
- [ ] Track Tested Claims
- [ ] Track Verified Resume Claims
- [ ] Track Partially Verified Resume Claims
- [ ] Track Unverified Resume Claims
- [ ] Track Contradicted Claims
- [ ] Track Skill Mastery
- [ ] Track Technology Depth
- [ ] Track Project Understanding
- [ ] Track Architecture Understanding
- [ ] Track Security Understanding
- [ ] Track Performance Understanding

## 8.3 Final Report

- [ ] Generate Improvement Areas
- [ ] Generate Interview Summary
- [ ] Generate Evidence-Based Summary
- [ ] Generate Skill Analysis
- [ ] Generate Knowledge Gap Analysis
- [ ] Generate Claim Verification Matrix
- [ ] Generate Project Understanding Analysis
- [ ] Generate Question/Answer History
- [ ] Generate Difficulty Progression
- [ ] Create Final Report API
- [ ] Mark Interview as COMPLETED
- [ ] Store Final Interview Results

## 8.4 Report Reliability

- [ ] Validate Report Against Stored Interview Data
- [ ] Prevent Unsupported Report Claims
- [ ] Preserve Evidence References
- [ ] Store Report Generation Metadata
- [ ] Version Report Generation Logic
- [ ] Handle Report Generation Failure

## 8.5 Testing

- [ ] Test Complete Interview Report
- [ ] Test Category-wise Analytics
- [ ] Test Claim Verification Analytics
- [ ] Test Knowledge Gap Analytics
- [ ] Test Strong Candidate Report
- [ ] Test Weak Candidate Report
- [ ] Test Mixed Candidate Report

## 8.6 Optional Voice Demonstrator

- [ ] Create Text-to-Speech Service
- [ ] Convert Generated Interview Questions to Speech
- [ ] Play AI Question Through Browser
- [ ] Add Start/Stop Voice Controls
- [ ] Add Question Playback State
- [ ] Add Loading State During Speech Generation
- [ ] Create Speech-to-Text Input
- [ ] Capture Candidate Voice Response
- [ ] Convert Candidate Speech to Text
- [ ] Send Transcribed Answer to Existing Evaluation Engine
- [ ] Connect Voice Answer → Answer Evaluation
- [ ] Connect Evaluation → Adaptive Question Controller
- [ ] Convert Adaptive Follow-up Question to Speech
- [ ] Test Complete Voice Interview Loop
- [ ] Handle Speech API Failure
- [ ] Handle Microphone Permission Failure
- [ ] Handle Transcription Failure

### Phase 8 Milestone

- [ ] Complete candidate intelligence report works
- [ ] Report is evidence-based
- [ ] Resume claims are categorized by verification state
- [ ] Knowledge gaps are clearly identified
- [ ] Optional voice demonstration works without affecting text interview


# Phase 9: Production Evaluation, Benchmarking & Ablation

## 9.1 Dataset

- [ ] Create Ground-Truth Resume Dataset
- [ ] Prepare Annotated Resumes
- [ ] Define Resume Diversity Criteria
- [ ] Include Different Resume Formats
- [ ] Include Different Sections
- [ ] Include Multi-Project Resumes
- [ ] Include Sparse Resumes
- [ ] Include Dense Resumes
- [ ] Define Expected Entities
- [ ] Define Expected Relationships
- [ ] Define Expected Resume Claims
- [ ] Define Expected Project Boundaries

## 9.2 Entity Evaluation

- [ ] Measure Entity Precision
- [ ] Measure Entity Recall
- [ ] Calculate Entity F1-Score
- [ ] Measure Entity Canonicalization Accuracy
- [ ] Measure Duplicate Entity Rate

## 9.3 Relationship Evaluation

- [ ] Measure Relationship Precision
- [ ] Measure Relationship Recall
- [ ] Calculate Relationship F1-Score
- [ ] Measure Relationship Evidence Accuracy
- [ ] Measure Relationship Context Accuracy
- [ ] Measure Project Boundary Accuracy
- [ ] Measure Phantom Project Rate
- [ ] Measure Unsupported Relationship Rate

## 9.4 Claim Evaluation

- [ ] Measure Resume Claim Extraction Accuracy
- [ ] Measure Claim Evidence Accuracy
- [ ] Measure Claim Verification Accuracy
- [ ] Measure Claim Contradiction Detection

## 9.5 RAG Evaluation

- [ ] Measure Retrieval Precision
- [ ] Measure Retrieval Recall
- [ ] Measure Context Relevance
- [ ] Measure Evidence Coverage
- [ ] Measure Retrieval Grounding
- [ ] Measure Reranker Effectiveness
- [ ] Measure Graph Retrieval Effectiveness
- [ ] Measure Vector Retrieval Effectiveness
- [ ] Measure Hybrid Retrieval Effectiveness

## 9.6 Question Evaluation

- [ ] Measure Question Relevance
- [ ] Measure Question Grounding
- [ ] Measure Question Technical Quality
- [ ] Measure Question Difficulty Accuracy
- [ ] Measure Question Duplication
- [ ] Measure Relationship Coverage
- [ ] Measure Claim Coverage
- [ ] Measure Project Coverage
- [ ] Measure Cross-Section Coverage

## 9.7 Adaptive Evaluation

- [ ] Measure Adaptive Follow-up Relevance
- [ ] Measure Knowledge-Gap Targeting
- [ ] Measure Difficulty Adaptation Accuracy
- [ ] Measure Repetition Rate
- [ ] Measure Topic Coverage
- [ ] Measure Claim Verification Coverage

## 9.8 Performance Evaluation

- [ ] Measure Resume Parsing Time
- [ ] Measure Entity Extraction Time
- [ ] Measure Relationship Extraction Time
- [ ] Measure Embedding Time
- [ ] Measure Vector Retrieval Time
- [ ] Measure Graph Retrieval Time
- [ ] Measure Reranking Time
- [ ] Measure RAG Retrieval Time
- [ ] Measure Question Generation Time
- [ ] Measure Answer Evaluation Time
- [ ] Calculate Average End-to-End Latency
- [ ] Measure Peak Memory Usage
- [ ] Measure LLM Token Usage
- [ ] Measure API Cost

## 9.9 Baselines

- [ ] Create Baseline System
- [ ] Implement Generic LLM Question Generation Baseline
- [ ] Implement Vector-Only RAG Baseline
- [ ] Implement Graph-Only Retrieval Baseline
- [ ] Compare Vector-Only vs Hybrid RAG
- [ ] Compare Generic vs Resume-Grounded Questions
- [ ] Compare Static vs Adaptive Interview

## 9.10 Ablation Studies

- [ ] Remove Graph Retrieval
- [ ] Remove Vector Retrieval
- [ ] Remove Reranking
- [ ] Remove Claim Grounding
- [ ] Remove Relationship Context
- [ ] Remove Project Context
- [ ] Remove Duplicate Detection
- [ ] Compare Performance Impact

## 9.11 Results

- [ ] Record Experimental Results
- [ ] Generate Evaluation Tables
- [ ] Generate Evaluation Graphs
- [ ] Calculate Confidence Intervals Where Appropriate
- [ ] Document Limitations
- [ ] Document Failure Cases
- [ ] Document Results
- [ ] Freeze Benchmark Dataset
- [ ] Freeze Evaluation Configuration

### Phase 9 Milestone

- [ ] IntervAI is quantitatively evaluated
- [ ] Extraction quality is measured
- [ ] Retrieval quality is measured
- [ ] Question quality is measured
- [ ] Adaptive behavior is measured
- [ ] System is compared against baselines
- [ ] Major components are validated through ablation


# Phase 10: Production Frontend & Interview Cockpit

## 10.1 Foundation

- [ ] Create React + Vite Project
- [ ] Configure Tailwind CSS
- [ ] Configure React Router
- [ ] Configure Axios
- [ ] Configure API Base URL
- [ ] Configure Environment Variables
- [ ] Create Base Layout
- [ ] Create Navigation
- [ ] Create Error Boundary
- [ ] Create Global Loading State
- [ ] Create Global Error State

## 10.2 Resume Interface

- [ ] Create Home Page
- [ ] Create Resume Upload Page
- [ ] Create File Dropzone
- [ ] Add File Validation
- [ ] Add Upload Progress
- [ ] Create Resume Processing Screen
- [ ] Display Processing Status
- [ ] Display Extracted Skills
- [ ] Display Extracted Technologies
- [ ] Display Extracted Projects
- [ ] Display Relationship Triplets
- [ ] Display Relationship Confidence
- [ ] Display Evidence
- [ ] Display Resume Claims
- [ ] Display Claim Verification State

## 10.3 Interview Cockpit

- [ ] Create Interview Start Page
- [ ] Create Interview Page
- [ ] Display Current Question
- [ ] Display Difficulty Level
- [ ] Display Question Category
- [ ] Display Resume Claim Being Tested
- [ ] Display Related Technology
- [ ] Display Related Project
- [ ] Create Answer Input
- [ ] Create Submit Answer Button
- [ ] Create Answer Loading State
- [ ] Create Evaluation Loading State
- [ ] Create Next Question Loading State
- [ ] Create Interview Progress
- [ ] Create Interview History
- [ ] Display Follow-Up Relationship
- [ ] Display Difficulty Changes

## 10.4 Evaluation Interface

- [ ] Create Evaluation Feedback View
- [ ] Display Score
- [ ] Display Strengths
- [ ] Display Weaknesses
- [ ] Display Missing Concepts
- [ ] Display Knowledge Gaps
- [ ] Display Feedback
- [ ] Display Tested Claim
- [ ] Display Evidence

## 10.5 Final Report

- [ ] Create Final Report Page
- [ ] Display Overall Score
- [ ] Display Category Scores
- [ ] Display Strengths
- [ ] Display Weaknesses
- [ ] Display Knowledge Gaps
- [ ] Display Verified Claims
- [ ] Display Partially Verified Claims
- [ ] Display Unverified Claims
- [ ] Display Contradicted Claims
- [ ] Display Question & Answer History
- [ ] Display Evaluation Feedback
- [ ] Display Improvement Areas
- [ ] Add Charts for Analytics
- [ ] Add Evidence Drill-Down
- [ ] Add Report Export

## 10.6 Frontend Reliability

- [ ] Implement API Error Handling
- [ ] Implement Retry UI
- [ ] Implement Request Cancellation
- [ ] Handle Session Expiration
- [ ] Handle Network Failure
- [ ] Handle Backend Failure
- [ ] Handle Empty States
- [ ] Handle Slow Requests
- [ ] Implement Responsive Design
- [ ] Accessibility Review
- [ ] UI Performance Optimization
- [ ] UI Polish

### Phase 10 Milestone

- [ ] Complete React frontend is connected to FastAPI
- [ ] Resume → Interview → Evaluation → Report works through UI
- [ ] Production error/loading states are handled


# Phase 11: Production Integration, Security & Final Testing

## 11.1 End-to-End Integration

- [ ] Connect Frontend to Backend
- [ ] Configure Production API URL
- [ ] Test Resume Upload → Database
- [ ] Test Resume → Text Extraction
- [ ] Test Resume → Entity Extraction
- [ ] Test Entity → Relationship Extraction
- [ ] Test Relationship → Knowledge Graph
- [ ] Test Knowledge Graph → RAG Retrieval
- [ ] Test RAG → Question Generation
- [ ] Test Question → Answer Submission
- [ ] Test Answer → Evaluation
- [ ] Test Evaluation → Adaptive Follow-up
- [ ] Test Interview → Final Report
- [ ] Test Complete End-to-End Workflow

## 11.2 Input & Failure Testing

- [ ] Handle Empty Resume
- [ ] Handle Corrupt PDF
- [ ] Handle Malicious/Abnormal PDF
- [ ] Handle Very Short Resume
- [ ] Handle Very Long Resume
- [ ] Handle Missing Resume Sections
- [ ] Handle Multi-Column Resume
- [ ] Handle Short Answers
- [ ] Handle Very Long Answers
- [ ] Handle Invalid Requests
- [ ] Handle Invalid IDs
- [ ] Handle API Errors
- [ ] Handle Database Errors
- [ ] Handle LLM/API Timeout
- [ ] Handle Embedding Failure
- [ ] Handle RAG Retrieval Failure
- [ ] Handle Vector Store Failure
- [ ] Handle Graph Retrieval Failure

## 11.3 Security

- [ ] Validate Uploaded File Type
- [ ] Validate Uploaded File Size
- [ ] Sanitize Uploaded Content
- [ ] Protect Against Path Traversal
- [ ] Protect Against Malicious PDF Content
- [ ] Protect Against Prompt Injection from Resume
- [ ] Treat Resume Text as Untrusted Input
- [ ] Validate LLM Structured Output
- [ ] Protect API Keys
- [ ] Remove Secrets from Source Code
- [ ] Configure Secure CORS
- [ ] Add Request Rate Limiting
- [ ] Add Abuse Protection
- [ ] Validate Database Inputs
- [ ] Prevent SQL Injection
- [ ] Configure Secure Headers
- [ ] Review Dependency Vulnerabilities

## 11.4 Observability

- [ ] Add Structured Logging
- [ ] Add Request IDs
- [ ] Add Pipeline IDs
- [ ] Add Error IDs
- [ ] Track Request Latency
- [ ] Track Database Latency
- [ ] Track RAG Latency
- [ ] Track LLM Latency
- [ ] Track Token Usage
- [ ] Track Retrieval Quality Metadata
- [ ] Track Failed Pipeline Stages
- [ ] Create Health Endpoint
- [ ] Create Readiness Endpoint
- [ ] Create Metrics Endpoint

## 11.5 Performance

- [ ] Optimize Slow Endpoints
- [ ] Optimize Database Queries
- [ ] Verify Database Indexes
- [ ] Add Appropriate Connection Pooling
- [ ] Add Caching Where Appropriate
- [ ] Optimize Embedding Generation
- [ ] Optimize Retrieval
- [ ] Optimize LLM Context Size
- [ ] Implement Request Timeouts
- [ ] Implement Service Timeouts
- [ ] Load Test Critical APIs

## 11.6 Code Quality

- [ ] Remove Debug Code
- [ ] Remove Temporary Scripts
- [ ] Remove Dead Code
- [ ] Remove Duplicate Logic
- [ ] Validate Type Hints
- [ ] Run Linter
- [ ] Run Formatter
- [ ] Run Unit Tests
- [ ] Run Integration Tests
- [ ] Run API Tests
- [ ] Run Regression Tests
- [ ] Code Review
- [ ] Verify API Responses
- [ ] Verify Frontend Error States

## 11.7 Documentation

- [ ] Update README
- [ ] Update API Documentation
- [ ] Add Setup Instructions
- [ ] Add Environment Variable Documentation
- [ ] Add Architecture Documentation
- [ ] Add Database Documentation
- [ ] Add RAG Documentation
- [ ] Add AI/NLP Pipeline Documentation
- [ ] Add Security Documentation

## 11.8 Release

- [ ] Final End-to-End Testing
- [ ] Final Security Review
- [ ] Final Performance Review
- [ ] Final Database Review
- [ ] Final API Review
- [ ] Final UI Review
- [ ] Code Freeze
- [ ] Final Git Commit
- [ ] Create Demo Release Tag
- [ ] Create Release Notes

### Phase 11 Milestone

- [ ] Complete IntervAI system works end-to-end
- [ ] Security controls are implemented
- [ ] Failure scenarios are handled
- [ ] Observability is available
- [ ] Performance is acceptable
- [ ] Codebase is release-ready


# Phase 12: Production Deployment

## 12.1 Infrastructure

- [ ] Prepare Production Environment
- [ ] Configure Production `.env`
- [ ] Configure Secret Management
- [ ] Configure Production PostgreSQL
- [ ] Configure Database Backups
- [ ] Configure Database Migrations
- [ ] Configure Connection Pooling
- [ ] Configure Production Logging

## 12.2 Backend Deployment

- [ ] Deploy FastAPI Backend
- [ ] Configure Production API Keys
- [ ] Configure LLM Provider
- [ ] Configure Embedding Provider
- [ ] Configure Vector Storage
- [ ] Verify Backend Health Endpoint
- [ ] Verify Readiness Endpoint
- [ ] Verify Production Database Connection
- [ ] Verify RAG Services
- [ ] Verify LLM Services

## 12.3 Frontend Deployment

- [ ] Deploy React Frontend
- [ ] Configure Frontend API URL
- [ ] Configure Production Environment Variables
- [ ] Test Frontend → Backend Communication
- [ ] Verify HTTPS
- [ ] Verify CORS

## 12.4 Production Testing

- [ ] Test Production Resume Upload
- [ ] Test Production Resume Processing
- [ ] Test Production Entity Extraction
- [ ] Test Production Relationship Extraction
- [ ] Test Production RAG Retrieval
- [ ] Test Production Question Generation
- [ ] Test Production Answer Evaluation
- [ ] Test Production Adaptive Interview
- [ ] Test Production Final Report
- [ ] Test Production Voice Demonstrator if Enabled

## 12.5 Reliability

- [ ] Configure Production Error Handling
- [ ] Configure Logging
- [ ] Configure Monitoring
- [ ] Configure Alerts
- [ ] Configure Health Checks
- [ ] Configure Database Backup Verification
- [ ] Configure Service Restart Policy
- [ ] Test Recovery from Backend Failure
- [ ] Test Recovery from Database Failure
- [ ] Test Recovery from LLM Failure
- [ ] Test Recovery from Vector Retrieval Failure

## 12.6 Security

- [ ] Verify Secrets Are Not Public
- [ ] Verify Production CORS
- [ ] Verify Rate Limits
- [ ] Verify File Upload Restrictions
- [ ] Verify API Authentication/Authorization if Enabled
- [ ] Verify HTTPS
- [ ] Verify Security Headers
- [ ] Verify Dependency Security

## 12.7 Final Verification

- [ ] Verify Deployment URLs
- [ ] Verify API Documentation
- [ ] Verify Production Logs
- [ ] Verify Metrics
- [ ] Verify Database Backups
- [ ] Run Production Smoke Tests
- [ ] Record Production Configuration
- [ ] Record Deployment Version
- [ ] Create Production Release Tag

### Phase 12 Milestone

- [ ] IntervAI is accessible through the deployed frontend
- [ ] Backend and database operate successfully in production
- [ ] Monitoring and recovery mechanisms are configured
- [ ] Production security checks pass


# Phase 13: Production Documentation, Research Validation & Project Presentation

## 13.1 Technical Documentation

- [ ] Finalize README
- [ ] Add Project Overview
- [ ] Add Problem Statement
- [ ] Add Proposed Solution
- [ ] Add System Architecture
- [ ] Add Detailed Component Architecture
- [ ] Add Technology Stack
- [ ] Add Database Schema
- [ ] Add ER Diagram
- [ ] Add API Documentation
- [ ] Add AI/NLP Pipeline Explanation
- [ ] Add Entity Extraction Explanation
- [ ] Add Relationship Extraction Explanation
- [ ] Add Knowledge Graph Explanation
- [ ] Add Hybrid RAG Explanation
- [ ] Add Question Generation Explanation
- [ ] Add Answer Evaluation Explanation
- [ ] Add Adaptive Interview Explanation
- [ ] Add Candidate Analytics Explanation

## 13.2 Research & Evaluation Documentation

- [ ] Add Evaluation Methodology
- [ ] Add Dataset Description
- [ ] Add Annotation Methodology
- [ ] Add Entity Evaluation Results
- [ ] Add Relationship Evaluation Results
- [ ] Add Claim Evaluation Results
- [ ] Add RAG Evaluation Results
- [ ] Add Question Evaluation Results
- [ ] Add Adaptive Interview Results
- [ ] Add Latency Results
- [ ] Add Baseline Comparison
- [ ] Add Ablation Results
- [ ] Add Failure Case Analysis
- [ ] Add Limitations
- [ ] Add Threats to Validity
- [ ] Add Benchmark Results

## 13.3 Demo Documentation

- [ ] Add Screenshots
- [ ] Add Demo Instructions
- [ ] Add Installation Instructions
- [ ] Add Environment Setup Instructions
- [ ] Add API Usage Examples
- [ ] Add Sample Resume
- [ ] Add Sample Interview
- [ ] Add Sample Evaluation
- [ ] Add Sample Final Report
- [ ] Add Troubleshooting Guide
- [ ] Add Future Scope

## 13.4 Project Presentation

- [ ] Prepare Project Presentation
- [ ] Prepare Project Demo
- [ ] Prepare Technical Explanation
- [ ] Prepare Architecture Diagram
- [ ] Prepare Database ER Diagram
- [ ] Prepare AI/NLP Pipeline Diagram
- [ ] Prepare RAG Pipeline Diagram
- [ ] Prepare Adaptive Interview Diagram
- [ ] Prepare Evaluation Results
- [ ] Prepare Benchmark Comparison
- [ ] Prepare Final Project Report
- [ ] Prepare Viva Questions & Answers
- [ ] Prepare Technical Defense for Design Decisions

## 13.5 Final Release

- [ ] Verify All Documentation
- [ ] Verify All Screenshots
- [ ] Verify Architecture Diagrams
- [ ] Verify Benchmark Results
- [ ] Verify Demo
- [ ] Verify Production Deployment
- [ ] Verify GitHub Repository
- [ ] Verify README Setup
- [ ] Verify No Secrets in Repository
- [ ] Create Final Release Tag
- [ ] Archive Final Benchmark Results
- [ ] Archive Final Configuration
- [ ] Archive Final Demo Build

### Phase 13 Milestone

- [ ] Complete project documentation is available
- [ ] Evaluation methodology and results are documented
- [ ] Project is ready for presentation
- [ ] Project is ready for technical evaluation
- [ ] Project is ready for demonstration
- [ ] Production release is documented


# Production Development Workflow

After completing each phase:

- [ ] Test functionality
- [ ] Test APIs using Postman
- [ ] Run automated tests
- [ ] Verify PostgreSQL data
- [ ] Verify logs and errors
- [ ] Verify provenance/evidence
- [ ] Fix edge cases
- [ ] Run regression tests
- [ ] Update documentation
- [ ] Review security implications
- [ ] Review performance implications
- [ ] Commit changes to GitHub
- [ ] Update `development.md`

## Definition of Done for Every Phase

- [ ] Feature implemented
- [ ] Feature tested
- [ ] Failure cases tested
- [ ] Data persistence verified
- [ ] API behavior verified
- [ ] Logs verified
- [ ] Security reviewed
- [ ] Performance reviewed
- [ ] Documentation updated
- [ ] Regression tests pass
- [ ] Git commit created


# Final Production Development Pipeline

Phase 1 → Project Setup & Database
↓
Phase 2 → Resume Ingestion
↓
Phase 3 → Entity Extraction
↓
Phase 4 → Relationship Extraction & Candidate Knowledge Graph
↓
Phase 5 → Hybrid Graph + Vector RAG & Grounded Question Generation
↓
Phase 6 → Evidence-Based Answer Evaluation
↓
Phase 7 → Adaptive Interview & Next-Best-Question Engine
↓
Phase 8 → Candidate Intelligence & Final Report
↓
Phase 9 → Evaluation, Benchmarking & Ablation
↓
Phase 10 → Production Frontend & Interview Cockpit
↓
Phase 11 → Integration, Security & Final Testing
↓
Phase 12 → Production Deployment
↓
Phase 13 → Documentation, Research Validation & Presentation


# Production Core Intelligence Pipeline

Resume PDF
↓
PDF Text Extraction
↓
Text Cleaning & Section Detection
↓
Transformer-Based Entity Extraction
↓
Entity Normalization
↓
Relationship Extraction
↓
Evidence Validation
↓
Candidate Knowledge Graph
↓
Resume Claim Intelligence
↓
Semantic Chunking
↓
Vector Index
↓
Graph Retrieval + Vector Retrieval + Lexical Retrieval
↓
Hybrid Result Fusion
↓
Reranking
↓
Evidence Selection
↓
Context Assembly
↓
Grounding Validation
↓
LLM Question Generation
↓
Question Quality Validation
↓
Interview Question
↓
Candidate Answer
↓
Evidence-Based Answer Evaluation
↓
Knowledge Gap Detection
↓
Candidate Knowledge State
↓
Next-Best-Question Engine
↓
Adaptive Follow-up
↓
Final Candidate Analytics
↓
Evidence-Based Final Report


# Production Core Database

- [ ] `resumes`
- [ ] `skills`
- [ ] `resume_skills`
- [ ] `relationships`
- [ ] `resume_chunks`
- [ ] `resume_embeddings`
- [ ] `resume_claims`
- [ ] `claim_evidence`
- [ ] `interview_sessions`
- [ ] `interview_questions`
- [ ] `interview_answers`
- [ ] `answer_evaluations`
- [ ] `knowledge_gaps`
- [ ] `candidate_knowledge_state`
- [ ] `retrieval_traces`
- [ ] `generation_metadata`
- [ ] `analytics`
- [ ] `reports`


# Production API Modules

- [ ] Resume Upload APIs
- [ ] Resume Processing APIs
- [ ] Entity Extraction APIs
- [ ] Relationship APIs
- [ ] Knowledge Graph APIs
- [ ] Resume Claim APIs
- [ ] RAG Retrieval APIs
- [ ] Question Generation APIs
- [ ] Interview Session APIs
- [ ] Answer Submission APIs
- [ ] Answer Evaluation APIs
- [ ] Adaptive Interview APIs
- [ ] Knowledge State APIs
- [ ] Analytics APIs
- [ ] Final Report APIs
- [ ] Health APIs
- [ ] Metrics APIs


# Production Success Criteria

- [ ] Resume information is converted into structured entities
- [ ] Relationships between technologies, projects and implementation claims are identified
- [ ] Every important relationship has evidence/provenance
- [ ] Phantom projects are prevented
- [ ] Unsupported relationships are rejected
- [ ] Candidate claims are extracted and grounded
- [ ] Resume context is retrieved using hybrid retrieval
- [ ] Questions are grounded in actual resume information
- [ ] Unsupported questions are rejected
- [ ] Duplicate questions are prevented
- [ ] Candidate answers are evaluated consistently
- [ ] Knowledge gaps are identified
- [ ] Interview difficulty adapts to candidate performance
- [ ] Next questions are selected using candidate knowledge state
- [ ] Repeated questions and topics are prevented
- [ ] Final report contains evidence-based insights
- [ ] Entity extraction is quantitatively evaluated
- [ ] Relationship extraction is quantitatively evaluated
- [ ] Claim extraction is quantitatively evaluated
- [ ] RAG retrieval is quantitatively evaluated
- [ ] Question generation is quantitatively evaluated
- [ ] Adaptive interviewing is quantitatively evaluated
- [ ] System performance is benchmarked against baselines
- [ ] Security controls are implemented
- [ ] Production monitoring is implemented
- [ ] Production failure recovery is tested
- [ ] Documentation is complete


# MVP Exclusions

The following features are NOT required for the first production version:

- [ ] AI Avatar
- [ ] Facial Emotion Detection
- [ ] Real-Time Video Interview
- [ ] Mobile Application
- [ ] Massive External Knowledge Graph
- [ ] Advanced Career Recommendation
- [ ] Job Matching
- [ ] Complex Project Recommendation
- [ ] Fine-Tuning BERT From Scratch

Voice interview remains optional and must not block the core text-based interview system.


# Future Features

- [ ] Voice-Based Interview
- [ ] Advanced Speech-to-Text
- [ ] Advanced Text-to-Speech
- [ ] AI Interview Avatar
- [ ] Emotion/Confidence Analysis
- [ ] Advanced Candidate Skill Graph
- [ ] Career Recommendation
- [ ] Job Matching
- [ ] Resume Improvement Suggestions
- [ ] Project Recommendation Engine
- [ ] Skill Gap → Project Recommendation
- [ ] Job Description → Customized Interview
- [ ] Multi-Round Interview Simulation
- [ ] Company-Specific Interview Modes
- [ ] Interview History & Long-Term Progress Tracking
