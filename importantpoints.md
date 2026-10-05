(CLICK ON EDIT TO SEE THESE PROPERLY)
                 
                 INTERVAI
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
   PostgreSQL 18             Alembic
   🗄️ ACTUAL DB              🔧 DB VERSION CONTROL
        │                       │
   Stores your data       Tracks DB changes
        │                       │
   resumes, skills,       Creates/updates tables
   questions, answers     through migrations



   SQLAlchemy = Python ↔ Database bridge


   pip install spacy installs spaCy, a Python library for Natural Language Processing (NLP).
  For our IntervAI project, spaCy can help us process the resume text after we extract it from the PDF.
  What can spaCy do?
  For example, if the resume contains:
  "Developed a Spring Boot backend using Java and PostgreSQL. Implemented JWT authentication and Redis caching."
