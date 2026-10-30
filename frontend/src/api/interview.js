import apiClient from './client';

/**
 * Lists all interview sessions belonging to the authenticated user.
 * GET /api/v1/sessions
 */
export async function listUserSessions() {
  const response = await apiClient.get('/sessions');
  return response.data;
}

/**
 * Starts or resumes an adaptive interview session for a resume.
 * POST /api/v1/sessions/start
 */
export async function startInterview({ resume_id, session_id, initial_difficulty = 3, session_metadata }) {
  const payload = {
    resume_id: resume_id || null,
    session_id: session_id || null,
    initial_difficulty: Number(initial_difficulty) || 3,
    session_metadata: session_metadata || null,
  };

  const response = await apiClient.post('/sessions/start', payload);
  return response.data;
}

/**
 * Retrieves full live interview state (asked questions, statuses, progress, gaps).
 * GET /api/v1/sessions/{session_id}/state
 */
export async function getInterviewState(sessionId) {
  const response = await apiClient.get(`/sessions/${sessionId}/state`);
  return response.data;
}

/**
 * Retrieves session metadata.
 * GET /api/v1/sessions/{session_id}
 */
export async function getSession(sessionId) {
  const response = await apiClient.get(`/sessions/${sessionId}`);
  return response.data;
}

/**
 * Selects and generates the next adaptive interview question.
 * POST /api/v1/sessions/{session_id}/next-question
 */
export async function getNextQuestion(sessionId, options = {}) {
  const payload = {
    preferred_category: options.preferred_category || null,
    force_strategy: options.force_strategy || null,
    max_consecutive_followups: options.max_consecutive_followups || null,
    request_idempotency_key: options.request_idempotency_key || null,
  };

  const response = await apiClient.post(`/sessions/${sessionId}/next-question`, payload, {
    timeout: 35000, // Bound next-question timeout to 35 seconds
  });
  return response.data;
}

/**
 * Submits a candidate answer for rubric evaluation.
 * POST /api/v1/evaluation/submit
 */
export async function submitAnswer({ question_id, session_id, candidate_answer }) {
  const payload = {
    question_id,
    session_id: session_id || null,
    candidate_answer: candidate_answer.trim(),
  };

  const response = await apiClient.post('/evaluation/submit', payload, {
    timeout: 35000, // Multi-dimensional evaluation with fast LLM
  });
  return response.data;
}

/**
 * Explicitly ends an interview session and triggers scorecard calculation.
 * POST /api/v1/sessions/{session_id}/end
 */
export async function endInterview(sessionId, reason = 'USER_COMPLETED', feedback_notes = null) {
  const payload = {
    reason,
    feedback_notes,
  };

  const response = await apiClient.post(`/sessions/${sessionId}/end`, payload);
  return response.data;
}

/**
 * Retrieves the calculated session scorecard.
 * GET /api/v1/sessions/{session_id}/scorecard
 */
export async function getSessionScorecard(sessionId) {
  const response = await apiClient.get(`/sessions/${sessionId}/scorecard`);
  return response.data;
}

/**
 * Retrieves the full Candidate Knowledge State for the session.
 * GET /api/v1/sessions/{session_id}/knowledge-state
 */
export async function getSessionKnowledgeState(sessionId) {
  const response = await apiClient.get(`/sessions/${sessionId}/knowledge-state`);
  return response.data;
}

/**
 * Retrieves structured knowledge gaps with severity ratings.
 * GET /api/v1/sessions/{session_id}/knowledge-state/gaps
 */
export async function getSessionKnowledgeGaps(sessionId) {
  const response = await apiClient.get(`/sessions/${sessionId}/knowledge-state/gaps`);
  return response.data;
}
