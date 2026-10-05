# IntervAI — Development Progress

Here’s the development roadmap for IntervAI.

---

# Phase 1: Project Setup & Database

- [x] Create FastAPI Project
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

# Phase 5: RAG & Question Generation Module

- [ ] Configure RAG Pipeline
- [ ] Select Embedding Model
- [ ] Configure Vector Storage
- [ ] Create Resume Context Retriever
- [ ] Create Resume Chunking Strategy
- [ ] Generate Resume Embeddings
- [ ] Store Resume Embeddings
- [ ] Retrieve Relevant Resume Sections
- [ ] Retrieve Related Entities
- [ ] Retrieve Relationship Triplets
- [ ] Combine Retrieved Context
- [ ] Create Question Generation Service
- [ ] Configure LLM
- [ ] Create Question Generation Prompt
- [ ] Create Fundamentals Question Logic
- [ ] Create Implementation Question Logic
- [ ] Create Architecture Question Logic
- [ ] Create Performance Question Logic
- [ ] Create Failure-Mode Question Logic
- [ ] Implement Difficulty Levels
  - [ ] Level 1 — Fundamentals
  - [ ] Level 2 — Application
  - [ ] Level 3 — Implementation
  - [ ] Level 4 — Architecture
  - [ ] Level 5 — Advanced Depth
- [ ] Implement Grounding Validation
- [ ] Validate Question Against Resume Context
- [ ] Reject Unsupported Resume Claims
- [ ] Implement Duplicate Question Detection
- [ ] Link Question to Resume Claim
- [ ] Link Question to Relationship
- [ ] Store Questions in PostgreSQL
- [ ] Create Question Generation API
- [ ] Test Question Generation in Postman
- [ ] Test Questions from Multiple Resume Sections

### Milestone:

- System generates personalized questions based on the candidate's actual resume
- Questions are grounded in retrieved resume context, entities, and relationships

---

# Phase 6: Answer Evaluation Module

- [ ] Create Answer Submission API
- [ ] Create Answer Evaluation Service
- [ ] Configure Evaluation Prompt
- [ ] Define Answer Evaluation Rubric
- [ ] Evaluate Technical Accuracy
- [ ] Evaluate Conceptual Understanding
- [ ] Evaluate Depth of Explanation
- [ ] Evaluate Implementation Knowledge
- [ ] Evaluate Architectural Reasoning
- [ ] Detect Missing Concepts
- [ ] Detect Knowledge Gaps
- [ ] Generate Answer Score
- [ ] Generate Evaluation Feedback
- [ ] Generate Strengths
- [ ] Generate Weaknesses
- [ ] Store Candidate Answer
- [ ] Store Score
- [ ] Store Feedback
- [ ] Store Knowledge Gaps
- [ ] Link Answer to Interview Question
- [ ] Update Interview Session Score
- [ ] Test Strong Answer
- [ ] Test Medium Answer
- [ ] Test Weak Answer
- [ ] Test Partially Correct Answer
- [ ] Test Irrelevant Answer

### Milestone:

- Candidate answers are evaluated
- System identifies strengths, weaknesses, and specific knowledge gaps

---

# Phase 7: Adaptive Interview Module

- [ ] Create Interview Session Service
- [ ] Create Start Interview API
- [ ] Create End Interview API
- [ ] Create Interview State Management
- [ ] Track Asked Questions
- [ ] Track Tested Skills
- [ ] Track Tested Relationships
- [ ] Track Knowledge Gaps
- [ ] Track Current Difficulty
- [ ] Track Interview Progress
- [ ] Implement Adaptive Question Logic

- [ ] Score < 5
  - [ ] Identify Weak Concept
  - [ ] Generate Targeted Follow-up

- [ ] Score 5–7.5
  - [ ] Ask Implementation Question
  - [ ] Ask Trade-off Question
  - [ ] Ask Reasoning Question

- [ ] Score > 7.5
  - [ ] Increase Difficulty
  - [ ] Ask Architecture Question
  - [ ] Ask Failure-Mode Question

- [ ] Implement Parent Question ID
- [ ] Implement Child Question Relationship
- [ ] Prevent Repeated Questions
- [ ] Prevent Repeated Topics
- [ ] Prevent Repeated Relationships
- [ ] Implement Difficulty Escalation
- [ ] Implement Weakness Probing
- [ ] Implement Strong-Answer Escalation
- [ ] Implement Partial-Answer Follow-up
- [ ] Implement Resume-Claim Verification Questions
- [ ] Test 5-Turn Interview Flow
- [ ] Test Weak-Answer Flow
- [ ] Test Strong-Answer Flow
- [ ] Test Mixed-Answer Flow

### Milestone:

- Interview dynamically changes based on candidate answers
- Strong answers increase difficulty
- Weak answers trigger targeted follow-ups
- Partial answers trigger deeper probing

---

# Phase 8: Candidate Analytics & Report

- [ ] Create Interview Analytics Service
- [ ] Calculate Overall Score
- [ ] Calculate Category Scores
- [ ] Calculate Backend Performance
- [ ] Calculate Database Performance
- [ ] Calculate Security Performance
- [ ] Calculate System Design Performance
- [ ] Calculate Technical Depth
- [ ] Identify Strengths
- [ ] Identify Weaknesses
- [ ] Identify Knowledge Gaps
- [ ] Track Tested Skills
- [ ] Track Tested Relationships
- [ ] Track Verified Resume Claims
- [ ] Track Partially Verified Claims
- [ ] Track Unverified Resume Claims
- [ ] Generate Improvement Areas
- [ ] Generate Interview Summary
- [ ] Create Final Report API
- [ ] Mark Interview as COMPLETED
- [ ] Store Final Interview Results
- [ ] Test Complete Interview Report
- [ ] Test Category-wise Analytics
- [ ] Test Claim Verification Analytics

### Milestone:

- System generates an evidence-based candidate performance report
- Report explains strengths, weaknesses, knowledge gaps, and verified claims

---

# Phase 9: Evaluation & Benchmarking

- [ ] Create Ground-Truth Resume Dataset
- [ ] Prepare 20 Annotated Resumes
- [ ] Define Expected Entities
- [ ] Define Expected Relationships
- [ ] Define Expected Resume Claims
- [ ] Measure Entity Precision
- [ ] Measure Entity Recall
- [ ] Calculate Entity F1-Score
- [ ] Measure Relationship Precision
- [ ] Measure Relationship Recall
- [ ] Calculate Relationship F1-Score
- [ ] Measure Resume Claim Extraction Accuracy
- [ ] Measure Question Relevance
- [ ] Measure Question Grounding
- [ ] Measure Question Technical Quality
- [ ] Measure Question Difficulty Accuracy
- [ ] Measure Question Duplication
- [ ] Measure Relationship Coverage
- [ ] Measure Adaptive Follow-up Relevance
- [ ] Measure Knowledge-Gap Targeting
- [ ] Measure Resume Parsing Time
- [ ] Measure Entity Extraction Time
- [ ] Measure Relationship Extraction Time
- [ ] Measure RAG Retrieval Time
- [ ] Measure Question Generation Time
- [ ] Measure Answer Evaluation Time
- [ ] Calculate Average End-to-End Latency
- [ ] Create Baseline System
- [ ] Implement Generic LLM Question Generation Baseline
- [ ] Compare Baseline vs IntervAI
- [ ] Record Experimental Results
- [ ] Generate Evaluation Tables
- [ ] Generate Evaluation Graphs
- [ ] Document Results

### Milestone:

- System performance is quantitatively evaluated
- Entity and relationship extraction are measured
- Question quality and adaptive behavior are measured
- IntervAI is compared against a simpler baseline

---

# Phase 10: Frontend Module

- [ ] Create React + Vite Project
- [ ] Configure Tailwind CSS
- [ ] Configure React Router
- [ ] Configure Axios
- [ ] Configure API Base URL
- [ ] Create Base Layout
- [ ] Create Navigation
- [ ] Create Home Page
- [ ] Create Resume Upload Page
- [ ] Create File Dropzone
- [ ] Add File Validation
- [ ] Add Upload Progress
- [ ] Create Resume Processing Screen
- [ ] Create Resume Intelligence View
- [ ] Display Extracted Skills
- [ ] Display Extracted Technologies
- [ ] Display Extracted Projects
- [ ] Display Relationship Triplets
- [ ] Display Relationship Confidence
- [ ] Create Interview Start Page
- [ ] Create Interview Page
- [ ] Display Current Question
- [ ] Display Difficulty Level
- [ ] Display Question Category
- [ ] Display Resume Claim Being Tested
- [ ] Display Related Technology
- [ ] Create Answer Input
- [ ] Create Submit Answer Button
- [ ] Create Answer Loading State
- [ ] Create Next Question Loading State
- [ ] Create Interview Progress
- [ ] Create Interview History
- [ ] Create Evaluation Feedback View
- [ ] Create Final Report Page
- [ ] Display Overall Score
- [ ] Display Category Scores
- [ ] Display Strengths
- [ ] Display Weaknesses
- [ ] Display Knowledge Gaps
- [ ] Display Verified Claims
- [ ] Display Unverified Claims
- [ ] Display Question & Answer History
- [ ] Display Evaluation Feedback
- [ ] Display Improvement Areas
- [ ] Add Charts for Analytics
- [ ] Implement Responsive Design
- [ ] UI Polish

### Milestone:

- Complete React frontend is connected to the FastAPI backend
- Complete Resume → Interview → Report workflow works through the UI

---

# Phase 11: Integration & Final Testing

- [ ] Connect Frontend to Backend
- [ ] Configure Production API URL
- [ ] Test Resume Upload → Database
- [ ] Test Resume → Text Extraction
- [ ] Test Resume → Entity Extraction
- [ ] Test Entity → Relationship Extraction
- [ ] Test Relationship → RAG Retrieval
- [ ] Test RAG → Question Generation
- [ ] Test Question → Answer Submission
- [ ] Test Answer → Evaluation
- [ ] Test Evaluation → Adaptive Follow-up
- [ ] Test Interview → Final Report
- [ ] Test Complete End-to-End Workflow
- [ ] Handle Empty Resume
- [ ] Handle Corrupt PDF
- [ ] Handle Very Short Resume
- [ ] Handle Missing Resume Sections
- [ ] Handle Large Resume
- [ ] Handle Short Answers
- [ ] Handle Very Long Answers
- [ ] Handle Invalid Requests
- [ ] Handle API Errors
- [ ] Handle Database Errors
- [ ] Handle LLM/API Timeout
- [ ] Handle RAG Retrieval Failure
- [ ] Add Backend Error Handling
- [ ] Add Input Validation
- [ ] Add Logging
- [ ] Add Request Validation
- [ ] Remove Debug Code
- [ ] Optimize Slow Endpoints
- [ ] Verify Database Indexes
- [ ] Verify API Responses
- [ ] Verify Frontend Error States
- [ ] Update README
- [ ] Update API Documentation
- [ ] Add Setup Instructions
- [ ] Add Environment Variable Documentation
- [ ] Add Architecture Documentation
- [ ] Final End-to-End Testing
- [ ] Code Review
- [ ] Code Freeze
- [ ] Final Git Commit
- [ ] Create Demo Release Tag

### Milestone:

- Complete IntervAI system works end-to-end
- Project is ready for demo, evaluation, and deployment

---

# Phase 12: Deployment

- [ ] Prepare Production Environment
- [ ] Configure Production `.env`
- [ ] Configure Production PostgreSQL
- [ ] Configure Database Migrations
- [ ] Deploy FastAPI Backend
- [ ] Verify Backend Health Endpoint
- [ ] Verify Production Database Connection
- [ ] Configure Production API Keys
- [ ] Deploy React Frontend
- [ ] Configure Frontend API URL
- [ ] Test Frontend → Backend Communication
- [ ] Test Production Resume Upload
- [ ] Test Production Interview Flow
- [ ] Test Production Final Report
- [ ] Configure CORS
- [ ] Configure Production Error Handling
- [ ] Configure Logging
- [ ] Verify Deployment URLs

### Milestone:

- IntervAI is accessible through a deployed frontend
- Backend and database operate successfully in production

---

# Phase 13: Documentation & Project Presentation

- [ ] Finalize README
- [ ] Add Project Overview
- [ ] Add Problem Statement
- [ ] Add Proposed Solution
- [ ] Add System Architecture
- [ ] Add Technology Stack
- [ ] Add Database Schema
- [ ] Add API Documentation
- [ ] Add AI/NLP Pipeline Explanation
- [ ] Add RAG Pipeline Explanation
- [ ] Add Adaptive Interview Logic
- [ ] Add Evaluation Methodology
- [ ] Add Benchmark Results
- [ ] Add Screenshots
- [ ] Add Demo Instructions
- [ ] Add Installation Instructions
- [ ] Add Environment Setup Instructions
- [ ] Add Future Scope
- [ ] Prepare Project Presentation
- [ ] Prepare Project Demo
- [ ] Prepare Technical Explanation
- [ ] Prepare Architecture Diagram
- [ ] Prepare Database ER Diagram
- [ ] Prepare Evaluation Results
- [ ] Prepare Final Project Report

### Milestone:

- Complete project documentation is available
- Project is ready for presentation, evaluation, and demonstration

---

# Development Workflow

After completing each phase:

- [ ] Test functionality
- [ ] Test APIs using Postman
- [ ] Verify PostgreSQL data
- [ ] Verify logs and errors
- [ ] Fix edge cases
- [ ] Update documentation
- [ ] Commit changes to GitHub
- [ ] Update `development.md`

Example Git workflow:

git add .
git commit -m "Phase 3: Implement entity extraction"
git push

---

# Final Development Pipeline

Phase 1 → Project Setup & Database

↓

Phase 2 → Resume Ingestion

↓

Phase 3 → Entity Extraction

↓

Phase 4 → Relationship Extraction

↓

Phase 5 → RAG & Question Generation

↓

Phase 6 → Answer Evaluation

↓

Phase 7 → Adaptive Interview

↓

Phase 8 → Candidate Analytics & Report

↓

Phase 9 → Evaluation & Benchmarking

↓

Phase 10 → Frontend

↓

Phase 11 → Integration & Final Testing

↓

Phase 12 → Deployment

↓

Phase 13 → Documentation & Presentation

---

# MVP

- [ ] Upload Resume
- [ ] Extract Resume Text
- [ ] Extract Entities
- [ ] Extract Relationships
- [ ] Retrieve Resume Context
- [ ] Generate Grounded Questions
- [ ] Answer Questions
- [ ] Evaluate Answers
- [ ] Generate Adaptive Follow-up Questions
- [ ] Complete Interview
- [ ] Generate Final Report

---

# Core Intelligence Pipeline

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
Candidate Knowledge Representation
↓
RAG Context Retrieval
↓
LLM Question Generation
↓
Grounding Validation
↓
Interview Question
↓
Candidate Answer
↓
Answer Evaluation
↓
Knowledge Gap Detection
↓
Adaptive Question Controller
↓
Next Question
↓
Final Candidate Analytics

---

# Core Database

- [ ] `resumes`
- [ ] `skills`
- [ ] `resume_skills`
- [ ] `relationships`
- [ ] `interview_sessions`
- [ ] `interview_questions`
- [ ] `interview_answers`

---

# API Modules

- [ ] Resume Upload APIs
- [ ] Resume Processing APIs
- [ ] Entity Extraction APIs
- [ ] Relationship APIs
- [ ] Question Generation APIs
- [ ] Interview Session APIs
- [ ] Answer Evaluation APIs
- [ ] Adaptive Interview APIs
- [ ] Analytics APIs
- [ ] Final Report APIs

---

# Project Success Criteria

- [ ] Resume information is converted into structured entities
- [ ] Relationships between technologies and implementation claims are identified
- [ ] Questions are grounded in actual resume information
- [ ] Unsupported claims are rejected or flagged
- [ ] Candidate answers are evaluated
- [ ] Knowledge gaps are identified
- [ ] Interview difficulty adapts to candidate performance
- [ ] Repeated questions are prevented
- [ ] Final report contains evidence-based insights
- [ ] Entity extraction is quantitatively evaluated
- [ ] Relationship extraction is quantitatively evaluated
- [ ] Question generation is quantitatively evaluated
- [ ] Adaptive interviewing is quantitatively evaluated
- [ ] System performance is benchmarked against a baseline

---

# MVP Exclusions

The following features are NOT required for the first version:

- [ ] Voice Interview
- [ ] Speech-to-Text
- [ ] Text-to-Speech
- [ ] AI Avatar
- [ ] Facial Emotion Detection
- [ ] Real-Time Video Interview
- [ ] Mobile Application
- [ ] Massive Knowledge Graph
- [ ] Advanced Career Recommendation
- [ ] Job Matching
- [ ] Complex Project Recommendation
- [ ] Fine-Tuning BERT From Scratch

---

# Future Features

- [ ] Voice-Based Interview
- [ ] Speech-to-Text
- [ ] Text-to-Speech
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
- [ ] Interview History & Progress Tracking
