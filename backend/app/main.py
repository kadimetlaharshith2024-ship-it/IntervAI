from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import auth, resumes, skills, relationships, rag, evaluation, sessions

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_startup_configuration()
    # Task 5: Warm FastEmbed and spaCy models at startup to eliminate runtime latency
    try:
        from app.services.rag.embedding_service import embedding_service
        embedding_service.embed_texts(["warmup"])
    except Exception as emb_err:
        import logging
        logging.getLogger(__name__).warning(f"FastEmbed warmup warning: {emb_err}")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        _ = nlp("warmup")
    except Exception as spacy_err:
        import logging
        logging.getLogger(__name__).warning(f"spaCy warmup warning: {spacy_err}")

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# API Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(resumes.router, prefix=f"{settings.API_V1_STR}/resumes", tags=["Resumes"])
app.include_router(skills.router, prefix=f"{settings.API_V1_STR}/skills", tags=["Skills & Entities"])
app.include_router(relationships.router, prefix=f"{settings.API_V1_STR}/relationships", tags=["Relationships"])
app.include_router(rag.router, prefix=f"{settings.API_V1_STR}/rag", tags=["RAG & Question Generation"])
app.include_router(evaluation.router, prefix=f"{settings.API_V1_STR}/evaluation", tags=["Answer Evaluation"])
app.include_router(sessions.router, prefix=f"{settings.API_V1_STR}/sessions", tags=["Interview Sessions"])

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}