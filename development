# IntervAI — Development Progress

Here’s the development roadmap for IntervAI.

---

# Phase 1: Project Setup & Database

⬜ Create FastAPI Project
⬜ Configure Python Virtual Environment
⬜ Configure Project Structure
⬜ Configure PostgreSQL
⬜ Configure SQLAlchemy
⬜ Configure `.env`
⬜ Configure Database Connection
⬜ Create Base Model
⬜ Create Database Session
⬜ Create Alembic Configuration
⬜ Create `Resume` Model
⬜ Create `Skill` Model
⬜ Create `ResumeSkill` Model
⬜ Create `Relationship` Model
⬜ Create `InterviewSession` Model
⬜ Create `InterviewQuestion` Model
⬜ Create `InterviewAnswer` Model
⬜ Add Primary Keys
⬜ Add Foreign Keys
⬜ Add Unique Constraints
⬜ Add Required Indexes
⬜ Create Initial Migration
⬜ Run Database Migration
⬜ Test PostgreSQL Connection

### Milestone:
- FastAPI project runs successfully
- PostgreSQL connection works
- All base tables are created successfully

---

# Phase 2: Resume Ingestion Module

⬜ Create Resume Upload Route
⬜ Create Resume Upload Service
⬜ Implement PDF Validation
⬜ Implement File Size Validation
⬜ Integrate `pdfplumber`
⬜ Extract Text from PDF
⬜ Clean Extracted Text
⬜ Normalize Whitespace
⬜ Remove Encoding Artifacts
⬜ Detect Resume Sections
⬜ Detect Skills Section
⬜ Detect Projects Section
⬜ Detect Experience Section
⬜ Detect Education Section
⬜ Save Resume Metadata
⬜ Save Raw Resume Text
⬜ Add Processing Status
⬜ Test Upload API in Postman
⬜ Test Multi-page Resume
⬜ Test Invalid PDF
⬜ Test Non-PDF File

### Milestone:
- Resume PDF can be uploaded
- Text is extracted correctly
- Resume information is stored in PostgreSQL

---

# Phase 3: Entity Extraction Module

⬜ Integrate Hugging Face Transformers
⬜ Configure Transformer Model
⬜ Create Entity Extraction Service
⬜ Extract Programming Languages
⬜ Extract Frameworks
⬜ Extract Libraries
⬜ Extract Databases
⬜ Extract Cloud Technologies
⬜ Extract Infrastructure Tools
⬜ Extract Project Names
⬜ Extract Quantifiable Metrics
⬜ Normalize Extracted Entities
⬜ Remove Duplicate Entities
⬜ Add Entity Confidence Score
⬜ Store Skills in `skills`
⬜ Store Resume-Skill Mapping in `resume_skills`
⬜ Create Entity Extraction API
⬜ Test Entity Extraction with Resume
⬜ Verify Extracted Entities Against Resume Text

### Milestone:
- Resume is converted from raw text into structured technical entities
- Extracted skills are stored in PostgreSQL

---

# Phase 4: Relationship Extraction Module

⬜ Define Relationship Types
   ├── USED_FOR
   ├── IMPLEMENTED_WITH
   ├── INTEGRATED_WITH
   ├── DEPLOYED_ON
   ├── OPTIMIZED_VIA
   └── BUILT_USING

⬜ Create Relationship Extraction Service
⬜ Detect Entity-to-Entity Relationships
⬜ Generate Relationship Triplets
⬜ Generate Relationship Confidence Score
⬜ Store Relationships in PostgreSQL
⬜ Create Relationship Query Service
⬜ Create Relationship API
⬜ Implement `GET /api/resumes/{id}/relationships`
⬜ Test Simple Relationships
⬜ Test Multiple Relationships
⬜ Test Project-Based Relationships
⬜ Verify Relationship Triplets Against Resume

### Example:

(JWT, USED_FOR, Authentication)

(Spring Security, IMPLEMENTS, JWT)

(Redis, USED_FOR, Caching)

(Spring Boot, INTEGRATED_WITH, PostgreSQL)

### Milestone:
- System understands relationships between technologies,
  projects, and implementation claims

---

# Phase 5: RAG & Question Generation Module

⬜ Configure RAG Pipeline
⬜ Create Resume Context Retriever
⬜ Retrieve Relevant Resume Sections
⬜ Retrieve Related Entities
⬜ Retrieve Relationship Triplets
⬜ Create Question Generation Service
⬜ Configure LLM
⬜ Create Question Generation Prompt
⬜ Create Fundamentals Question Logic
⬜ Create Implementation Question Logic
⬜ Create Architecture Question Logic
⬜ Create Performance Question Logic
⬜ Create Failure-Mode Question Logic
⬜ Implement Difficulty Levels
   ├── Level 1
   ├── Level 2
   ├── Level 3
   ├── Level 4
   └── Level 5

⬜ Implement Grounding Validation
⬜ Reject Unsupported Resume Claims
⬜ Implement Duplicate Question Detection
⬜ Store Questions in PostgreSQL
⬜ Link Question to Resume Claim
⬜ Link Question to Relationship
⬜ Create Question Generation API
⬜ Test Question Generation in Postman

### Milestone:
- System generates personalized questions based on the
  candidate's actual resume
- Questions are grounded in extracted entities and relationships

---

# Phase 6: Answer Evaluation Module

⬜ Create Answer Submission API
⬜ Create Answer Evaluation Service
⬜ Configure Evaluation Prompt
⬜ Evaluate Technical Accuracy
⬜ Evaluate Conceptual Understanding
⬜ Evaluate Depth
⬜ Evaluate Implementation Knowledge
⬜ Detect Missing Concepts
⬜ Detect Knowledge Gaps
⬜ Generate Score
⬜ Generate Feedback
⬜ Store Candidate Answer
⬜ Store Score
⬜ Store Feedback
⬜ Store Knowledge Gaps
⬜ Update Interview Session Score
⬜ Test Strong Answer
⬜ Test Medium Answer
⬜ Test Weak Answer

### Milestone:
- Candidate answers are evaluated
- System identifies strengths, weaknesses, and knowledge gaps

---

# Phase 7: Adaptive Interview Module

⬜ Create Interview Session Service
⬜ Create Start Interview API
⬜ Create End Interview API
⬜ Create Interview State Management
⬜ Track Asked Questions
⬜ Track Tested Skills
⬜ Track Tested Relationships
⬜ Track Knowledge Gaps
⬜ Track Current Difficulty
⬜ Implement Adaptive Question Logic

⬜ Score < 5
   └── Ask targeted follow-up

⬜ Score 5–7.5
   └── Ask implementation/trade-off question

⬜ Score > 7.5
   └── Increase difficulty

⬜ Implement Parent Question ID
⬜ Implement Child Question Relationship
⬜ Prevent Repeated Questions
⬜ Prevent Repeated Topics
⬜ Implement Difficulty Escalation
⬜ Implement Weakness Probing
⬜ Implement Strong-Answer Escalation
⬜ Test 5-Turn Interview Flow

### Milestone:
- Interview dynamically changes based on candidate answers
- Strong answers increase difficulty
- Weak answers trigger targeted follow-ups

---

# Phase 8: Candidate Analytics & Report

⬜ Create Interview Analytics Service
⬜ Calculate Overall Score
⬜ Calculate Category Scores
⬜ Calculate Backend Performance
⬜ Calculate Database Performance
⬜ Calculate Security Performance
⬜ Calculate System Design Performance
⬜ Identify Strengths
⬜ Identify Weaknesses
⬜ Identify Knowledge Gaps
⬜ Track Verified Resume Claims
⬜ Track Unverified Resume Claims
⬜ Generate Improvement Areas
⬜ Create Final Report API
⬜ Mark Interview as COMPLETED
⬜ Test Complete Interview Report

### Milestone:
- System generates an evidence-based candidate performance report

---

# Phase 9: Evaluation & Benchmarking

⬜ Create Ground-Truth Resume Dataset
⬜ Prepare 20 Annotated Resumes
⬜ Define Expected Entities
⬜ Define Expected Relationships
⬜ Measure Entity Precision
⬜ Measure Entity Recall
⬜ Calculate Entity F1-Score
⬜ Measure Relationship Precision
⬜ Measure Relationship Recall
⬜ Calculate Relationship F1-Score
⬜ Measure Question Relevance
⬜ Measure Question Grounding
⬜ Measure Question Duplication
⬜ Measure Adaptive Follow-up Relevance
⬜ Measure Resume Parsing Time
⬜ Measure Entity Extraction Time
⬜ Measure Relationship Extraction Time
⬜ Measure Question Generation Time
⬜ Measure Answer Evaluation Time

⬜ Create Baseline System
   └── Generic LLM Question Generator

⬜ Compare Baseline vs IntervAI
⬜ Record Experimental Results
⬜ Document Results

### Milestone:
- System performance is quantitatively evaluated
- Proposed pipeline is compared against a simpler baseline

---

# Phase 10: Frontend Module

⬜ Create React + Vite Project
⬜ Configure Tailwind CSS
⬜ Configure React Router
⬜ Configure Axios
⬜ Create Base Layout
⬜ Create Login Page
⬜ Create Resume Upload Page
⬜ Create Resume Processing Screen
⬜ Create Resume Intelligence View
⬜ Display Extracted Skills
⬜ Display Extracted Projects
⬜ Display Relationship Triplets
⬜ Create Interview Page
⬜ Display Current Question
⬜ Display Difficulty Level
⬜ Display Resume Claim Being Tested
⬜ Create Answer Input
⬜ Create Submit Answer Button
⬜ Create Interview Loading State
⬜ Create Interview Progress
⬜ Create Final Report Page
⬜ Display Overall Score
⬜ Display Category Scores
⬜ Display Strengths
⬜ Display Weaknesses
⬜ Display Knowledge Gaps
⬜ Display Verified Claims
⬜ Display Unverified Claims
⬜ Display Question & Answer History
⬜ Display Evaluation Feedback
⬜ Implement Responsive Design
⬜ UI Polish

### Milestone:
- Complete React frontend is connected to the FastAPI backend
- Complete Resume → Interview → Report workflow works through UI

---

# Phase 11: Integration & Final Testing

⬜ Connect Frontend to Backend
⬜ Test Resume Upload → Database
⬜ Test Resume → Entity Extraction
⬜ Test Entity → Relationship Extraction
⬜ Test Relationship → Question Generation
⬜ Test Question → Answer Evaluation
⬜ Test Answer → Adaptive Follow-up
⬜ Test Interview → Final Report

⬜ Handle Empty Resume
⬜ Handle Corrupt PDF
⬜ Handle Very Short Resume
⬜ Handle Missing Resume Sections
⬜ Handle Short Answers
⬜ Handle Invalid Requests
⬜ Handle API Errors
⬜ Handle Database Errors
⬜ Handle LLM/API Timeout
⬜ Add Backend Error Handling
⬜ Add Input Validation
⬜ Add Logging
⬜ Remove Debug Code
⬜ Update README
⬜ Update API Documentation
⬜ Final End-to-End Testing
⬜ Code Freeze
⬜ Final Git Commit
⬜ Create Demo Release Tag

### Milestone:
- Complete IntervAI system works end-to-end
- Project is ready for demo, evaluation, and deployment

---

# FINAL DEVELOPMENT PIPELINE

Phase 1
Project + Database
        ↓
Phase 2
Resume Processing
        ↓
Phase 3
Entity Extraction
        ↓
Phase 4
Relationship Extraction
        ↓
Phase 5
RAG + Question Generation
        ↓
Phase 6
Answer Evaluation
        ↓
Phase 7
Adaptive Interview
        ↓
Phase 8
Candidate Analytics
        ↓
Phase 9
Evaluation & Benchmarking
        ↓
Phase 10
React Frontend
        ↓
Phase 11
Integration & Final Testing

---

# DEVELOPMENT RULE

After completing every phase:

⬜ Test functionality
⬜ Test API using Postman
⬜ Verify PostgreSQL data
⬜ Fix edge cases
⬜ Commit changes to GitHub
⬜ Update this DEVELOPMENT_PROGRESS.md

Example:

git add .
git commit -m "Phase 3: Implement entity extraction"
git push

---

# PROJECT MVP

The first working version must support:

⬜ Upload Resume
⬜ Extract Resume Text
⬜ Extract Entities
⬜ Extract Relationships
⬜ Generate Grounded Questions
⬜ Answer Questions
⬜ Evaluate Answers
⬜ Generate Adaptive Follow-up Questions
⬜ Complete Interview
⬜ Generate Final Report

---

# OPTIONAL / FUTURE FEATURES

⬜ Voice Interview
⬜ Speech-to-Text
⬜ Text-to-Speech
⬜ AI Avatar
⬜ Emotion Detection
⬜ Advanced Skill Graph
⬜ Career Recommendation
⬜ Project Recommendation
⬜ Job Matching
⬜ Resume Improvement Suggestions
