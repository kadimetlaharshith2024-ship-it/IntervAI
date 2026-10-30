from datetime import timedelta
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.core.database import Base, get_db
from app.core.security import get_password_hash, verify_password, create_access_token
from app.models.user import User
from app.models.resume import Resume
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.main import app


@pytest.fixture(scope="function")
def auth_db():
    """Provides an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def auth_client(auth_db: Session):
    """TestClient configured with dependency overrides for the test DB."""
    def _override_get_db():
        try:
            yield auth_db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def create_test_user(auth_db: Session, email: str, password: str, full_name: str) -> User:
    user = User(
        id=uuid4(),
        email=email.strip().lower(),
        password_hash=get_password_hash(password),
        full_name=full_name
    )
    auth_db.add(user)
    auth_db.commit()
    auth_db.refresh(user)
    return user


def get_auth_headers_for_user(user: User) -> dict:
    token = create_access_token(data={"sub": str(user.id), "email": user.email})
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# 1. REGISTRATION TESTS
# ==============================================================================

def test_register_user_success(auth_client: TestClient, auth_db: Session):
    """Successfully registers a new user with valid credentials."""
    resp = auth_client.post(
        "/api/v1/auth/register",
        json={
            "email": "Alex.Dev@Example.COM",
            "password": "SecurePassword123!",
            "full_name": "Alex Dev"
        }
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["email"] == "alex.dev@example.com"  # Normalized lowercase
    assert data["full_name"] == "Alex Dev"
    assert "password" not in data
    assert "password_hash" not in data

    # Verify database persistence and password hashing
    user_in_db = auth_db.query(User).filter(User.email == "alex.dev@example.com").first()
    assert user_in_db is not None
    assert user_in_db.password_hash != "SecurePassword123!"
    assert verify_password("SecurePassword123!", user_in_db.password_hash)


def test_register_duplicate_email_rejected(auth_client: TestClient, auth_db: Session):
    """Prevents duplicate registrations with the same or case-variant email."""
    create_test_user(auth_db, "user@example.com", "Password123!", "Existing User")

    # Exact duplicate
    resp1 = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "Password123!", "full_name": "New User"}
    )
    assert resp1.status_code == 409
    assert "already exists" in resp1.json()["detail"].lower()

    # Case-insensitive duplicate
    resp2 = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "USER@Example.COM", "password": "Password123!", "full_name": "New User"}
    )
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"].lower()


def test_register_validation_errors(auth_client: TestClient):
    """Rejects malformed input: short passwords, invalid emails, empty names."""
    # Short password
    r1 = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "short@example.com", "password": "short", "full_name": "Short Pass"}
    )
    assert r1.status_code == 422

    # Invalid email
    r2 = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "ValidPassword123!", "full_name": "Bad Email"}
    )
    assert r2.status_code == 422

    # Blank full name
    r3 = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "blank@example.com", "password": "ValidPassword123!", "full_name": "   "}
    )
    assert r3.status_code == 422


# ==============================================================================
# 2. LOGIN TESTS
# ==============================================================================

def test_login_success(auth_client: TestClient, auth_db: Session):
    """Logs in with valid credentials and receives JWT and user info."""
    create_test_user(auth_db, "candidate@example.com", "StrongPassword123!", "Candidate One")

    resp = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "Candidate@Example.COM", "password": "StrongPassword123!"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "candidate@example.com"
    assert data["user"]["full_name"] == "Candidate One"
    assert "password_hash" not in data["user"]


def test_login_invalid_credentials(auth_client: TestClient, auth_db: Session):
    """Rejects invalid credentials with HTTP 401."""
    create_test_user(auth_db, "user@example.com", "CorrectPassword123!", "Valid User")

    # Wrong password
    r1 = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "WrongPassword!"}
    )
    assert r1.status_code == 401

    # Nonexistent user
    r2 = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@example.com", "password": "AnyPassword123!"}
    )
    assert r2.status_code == 401


# ==============================================================================
# 3. GET /AUTH/ME AND JWT TOKEN VALIDATION
# ==============================================================================

def test_auth_me_with_valid_token(auth_client: TestClient, auth_db: Session):
    """GET /auth/me returns authenticated user details when a valid token is provided."""
    user = create_test_user(auth_db, "me@example.com", "Password123!", "Me User")
    headers = get_auth_headers_for_user(user)

    resp = auth_client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(user.id)
    assert data["email"] == "me@example.com"
    assert data["full_name"] == "Me User"


def test_auth_me_without_token(auth_client: TestClient):
    """GET /auth/me returns 401 Unauthorized when unauthenticated."""
    resp = auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_auth_me_with_expired_token(auth_client: TestClient, auth_db: Session):
    """GET /auth/me returns 401 with explicit expired message for expired JWTs."""
    user = create_test_user(auth_db, "expired@example.com", "Password123!", "Expired User")
    expired_token = create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=timedelta(minutes=-10)  # Expired in past
    )
    headers = {"Authorization": f"Bearer {expired_token}"}

    resp = auth_client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"].lower()


def test_auth_me_with_invalid_or_tampered_token(auth_client: TestClient):
    """GET /auth/me returns 401 for tampered or invalid tokens."""
    # Malformed token
    r1 = auth_client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
    assert r1.status_code == 401

    # Tampered signature
    valid_format = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3OC0xMjM0LTUyMzQtYTIzNC0xMjM0NTY3ODkwYWIiLCJlbWFpbCI6ImZha2VAZXhhbXBsZS5jb20iLCJleHAiOjk5OTk5OTk5OTksImlhdCI6MTYwMDAwMDAwMH0.tampered_signature_here"
    r2 = auth_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {valid_format}"})
    assert r2.status_code == 401


# ==============================================================================
# 4. CROSS-USER OWNERSHIP ISOLATION TESTS
# ==============================================================================

@pytest.fixture
def multi_user_setup(auth_db: Session):
    """Creates User A and User B with dedicated resumes and sessions."""
    user_a = create_test_user(auth_db, "user.a@example.com", "PasswordUserA123!", "User Alpha")
    user_b = create_test_user(auth_db, "user.b@example.com", "PasswordUserB123!", "User Beta")

    resume_a = Resume(
        id=uuid4(),
        user_id=user_a.id,
        filename="resume_a.pdf",
        raw_text="Alpha Resume text with Python, PostgreSQL, and Redis experience."
    )
    resume_b = Resume(
        id=uuid4(),
        user_id=user_b.id,
        filename="resume_b.pdf",
        raw_text="Beta Resume text with Java and Spring experience."
    )
    auth_db.add_all([resume_a, resume_b])
    auth_db.flush()

    session_a = InterviewSession(
        id=uuid4(),
        user_id=user_a.id,
        resume_id=resume_a.id,
        status="IN_PROGRESS",
        current_difficulty=3,
        session_state={"tested_skills": []}
    )
    session_b = InterviewSession(
        id=uuid4(),
        user_id=user_b.id,
        resume_id=resume_b.id,
        status="IN_PROGRESS",
        current_difficulty=3,
        session_state={"tested_skills": []}
    )
    auth_db.add_all([session_a, session_b])
    auth_db.flush()

    question_a = InterviewQuestion(
        id=uuid4(),
        session_id=session_a.id,
        resume_id=resume_a.id,
        question_text="How did you implement caching with Redis in your project?",
        category="Architecture",
        difficulty=3
    )
    auth_db.add(question_a)
    auth_db.flush()

    answer_a = InterviewAnswer(
        id=uuid4(),
        question_id=question_a.id,
        session_id=session_a.id,
        candidate_answer="I used cache-aside pattern with 60 second TTL.",
        status="EVALUATED",
        overall_score=85.0
    )
    auth_db.add(answer_a)
    auth_db.commit()

    return {
        "user_a": user_a,
        "user_b": user_b,
        "headers_a": get_auth_headers_for_user(user_a),
        "headers_b": get_auth_headers_for_user(user_b),
        "resume_a": resume_a,
        "resume_b": resume_b,
        "session_a": session_a,
        "session_b": session_b,
        "question_a": question_a,
        "answer_a": answer_a,
    }


def test_user_b_cannot_access_user_a_resume(auth_client: TestClient, multi_user_setup):
    """User B cannot fetch User A's resume (returns HTTP 403)."""
    resume_a = multi_user_setup["resume_a"]
    headers_b = multi_user_setup["headers_b"]

    resp = auth_client.get(f"/api/v1/resumes/{resume_a.id}", headers=headers_b)
    assert resp.status_code == 403


def test_user_b_cannot_access_user_a_session(auth_client: TestClient, multi_user_setup):
    """User B cannot fetch User A's interview session metadata (returns HTTP 403)."""
    session_a = multi_user_setup["session_a"]
    headers_b = multi_user_setup["headers_b"]

    resp = auth_client.get(f"/api/v1/sessions/{session_a.id}", headers=headers_b)
    assert resp.status_code == 403


def test_user_b_cannot_start_interview_with_user_a_resume(auth_client: TestClient, multi_user_setup):
    """User B cannot start an interview session using User A's resume (returns HTTP 403)."""
    resume_a = multi_user_setup["resume_a"]
    headers_b = multi_user_setup["headers_b"]

    # 1. via POST /api/v1/sessions
    r1 = auth_client.post(
        "/api/v1/sessions",
        json={"resume_id": str(resume_a.id), "initial_difficulty": 3},
        headers=headers_b
    )
    assert r1.status_code == 403

    # 2. via POST /api/v1/sessions/start
    r2 = auth_client.post(
        "/api/v1/sessions/start",
        json={"resume_id": str(resume_a.id), "initial_difficulty": 3},
        headers=headers_b
    )
    assert r2.status_code == 403

    # 3. via POST /api/v1/resumes/{resume_id}/sessions
    r3 = auth_client.post(
        f"/api/v1/resumes/{resume_a.id}/sessions",
        headers=headers_b
    )
    assert r3.status_code == 403


def test_user_b_cannot_access_user_a_session_questions_or_scorecard(auth_client: TestClient, multi_user_setup):
    """User B cannot access questions or scorecards of User A's session."""
    session_a = multi_user_setup["session_a"]
    headers_b = multi_user_setup["headers_b"]

    # Session questions
    r1 = auth_client.get(f"/api/v1/sessions/{session_a.id}/questions", headers=headers_b)
    assert r1.status_code == 403

    # Scorecard via sessions endpoint
    r2 = auth_client.get(f"/api/v1/sessions/{session_a.id}/scorecard", headers=headers_b)
    assert r2.status_code == 403

    # Scorecard via evaluation endpoint
    r3 = auth_client.get(f"/api/v1/evaluation/session/{session_a.id}/scorecard", headers=headers_b)
    assert r3.status_code == 403


def test_user_b_cannot_access_user_a_knowledge_state(auth_client: TestClient, multi_user_setup):
    """User B cannot access knowledge-state operations for User A's session."""
    session_a = multi_user_setup["session_a"]
    headers_b = multi_user_setup["headers_b"]

    r1 = auth_client.get(f"/api/v1/sessions/{session_a.id}/knowledge-state", headers=headers_b)
    assert r1.status_code == 403

    r2 = auth_client.get(f"/api/v1/sessions/{session_a.id}/knowledge-state/topics", headers=headers_b)
    assert r2.status_code == 403

    r3 = auth_client.get(f"/api/v1/sessions/{session_a.id}/knowledge-state/gaps", headers=headers_b)
    assert r3.status_code == 403


def test_user_b_cannot_access_user_a_answers_or_history(auth_client: TestClient, multi_user_setup):
    """User B cannot access or evaluate answers for User A's question/session."""
    session_a = multi_user_setup["session_a"]
    question_a = multi_user_setup["question_a"]
    answer_a = multi_user_setup["answer_a"]
    headers_b = multi_user_setup["headers_b"]

    # History
    r1 = auth_client.get(
        f"/api/v1/evaluation/question/{question_a.id}/history?session_id={session_a.id}",
        headers=headers_b
    )
    assert r1.status_code == 403

    # Direct answer lookup
    r2 = auth_client.get(f"/api/v1/evaluation/{answer_a.id}", headers=headers_b)
    assert r2.status_code == 403

    # Answer submission attempt on User A's question/session
    r3 = auth_client.post(
        "/api/v1/evaluation/submit",
        json={
            "question_id": str(question_a.id),
            "session_id": str(session_a.id),
            "candidate_answer": "Unauthorized candidate attempt."
        },
        headers=headers_b
    )
    assert r3.status_code == 403


# ==============================================================================
# 5. USER-SCOPED RESOURCE LISTINGS
# ==============================================================================

def test_user_scoped_resume_listing(auth_client: TestClient, multi_user_setup):
    """GET /api/v1/resumes lists only the authenticated user's resumes."""
    headers_a = multi_user_setup["headers_a"]
    headers_b = multi_user_setup["headers_b"]
    resume_a = multi_user_setup["resume_a"]
    resume_b = multi_user_setup["resume_b"]

    # User A listing
    resp_a = auth_client.get("/api/v1/resumes", headers=headers_a)
    assert resp_a.status_code == 200
    resumes_a = resp_a.json()
    assert len(resumes_a) == 1
    assert resumes_a[0]["id"] == str(resume_a.id)

    # User B listing
    resp_b = auth_client.get("/api/v1/resumes", headers=headers_b)
    assert resp_b.status_code == 200
    resumes_b = resp_b.json()
    assert len(resumes_b) == 1
    assert resumes_b[0]["id"] == str(resume_b.id)

    # Unauthenticated listing must return 401
    resp_unauth = auth_client.get("/api/v1/resumes")
    assert resp_unauth.status_code == 401


def test_user_scoped_session_listing(auth_client: TestClient, multi_user_setup):
    """GET /api/v1/sessions lists only the authenticated user's sessions."""
    headers_a = multi_user_setup["headers_a"]
    headers_b = multi_user_setup["headers_b"]
    session_a = multi_user_setup["session_a"]
    session_b = multi_user_setup["session_b"]

    # User A listing
    resp_a = auth_client.get("/api/v1/sessions", headers=headers_a)
    assert resp_a.status_code == 200
    sessions_a = resp_a.json()
    assert len(sessions_a) == 1
    assert sessions_a[0]["id"] == str(session_a.id)

    # User B listing
    resp_b = auth_client.get("/api/v1/sessions", headers=headers_b)
    assert resp_b.status_code == 200
    sessions_b = resp_b.json()
    assert len(sessions_b) == 1
    assert sessions_b[0]["id"] == str(session_b.id)

    # Unauthenticated listing must return 401
    resp_unauth = auth_client.get("/api/v1/sessions")
    assert resp_unauth.status_code == 401


# ==============================================================================
# 6. HISTORICAL / LEGACY UNOWNED RECORDS COMPATIBILITY
# ==============================================================================

def test_legacy_unowned_records_isolated_from_authenticated_users(auth_client: TestClient, auth_db: Session):
    """
    Historical records with user_id=None must not be exposed to arbitrary authenticated users,
    while remaining accessible to unauthenticated callers for Phase 1-7 test suite compatibility.
    """
    user = create_test_user(auth_db, "user@example.com", "Password123!", "Some User")
    headers = get_auth_headers_for_user(user)

    legacy_resume = Resume(
        id=uuid4(),
        user_id=None,  # Legacy record
        filename="legacy.pdf",
        raw_text="Legacy resume text."
    )
    auth_db.add(legacy_resume)
    auth_db.flush()

    legacy_session = InterviewSession(
        id=uuid4(),
        user_id=None,
        resume_id=legacy_resume.id,
        status="IN_PROGRESS"
    )
    auth_db.add(legacy_session)
    auth_db.commit()

    # Authenticated user attempting to access legacy records -> 403 Forbidden
    r_auth_res = auth_client.get(f"/api/v1/resumes/{legacy_resume.id}", headers=headers)
    assert r_auth_res.status_code == 403

    r_auth_sess = auth_client.get(f"/api/v1/sessions/{legacy_session.id}", headers=headers)
    assert r_auth_sess.status_code == 403

    # Unauthenticated caller (legacy test compatibility) -> 200 OK
    r_unauth_res = auth_client.get(f"/api/v1/resumes/{legacy_resume.id}")
    assert r_unauth_res.status_code == 200

    r_unauth_sess = auth_client.get(f"/api/v1/sessions/{legacy_session.id}")
    assert r_unauth_sess.status_code == 200
