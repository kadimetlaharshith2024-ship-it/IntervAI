import apiClient from './client';

/**
 * Lists all resumes belonging to the authenticated user.
 * GET /api/v1/resumes
 */
export async function listResumes() {
  const response = await apiClient.get('/resumes');
  return response.data;
}

/**
 * Retrieves a single resume by its UUID after permission check.
 * GET /api/v1/resumes/{resume_id}
 */
export async function getResume(resumeId) {
  const response = await apiClient.get(`/resumes/${resumeId}`);
  return response.data;
}

/**
 * Uploads a PDF resume using multipart/form-data with field 'file'.
 * POST /api/v1/resumes/upload
 */
export async function uploadResume(file, onUploadProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/resumes/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 30000,
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onUploadProgress(percentCompleted);
      }
    },
  });

  return response.data;
}

/**
 * Triggers entity extraction for an uploaded resume.
 * POST /api/v1/skills/{resume_id}/extract-entities
 */
export async function extractEntities(resumeId) {
  const response = await apiClient.post(`/skills/${resumeId}/extract-entities`, null, {
    timeout: 30000,
  });
  return response.data;
}

/**
 * Retrieves previously extracted entities for a resume.
 * GET /api/v1/skills/{resume_id}
 */
export async function getEntities(resumeId) {
  const response = await apiClient.get(`/skills/${resumeId}`, {
    timeout: 15000,
  });
  return response.data;
}

/**
 * Triggers relationship extraction for a resume.
 * POST /api/v1/relationships/{resume_id}/extract-relationships?mode=advanced
 */
export async function extractRelationships(resumeId, mode = 'advanced') {
  const response = await apiClient.post(
    `/relationships/${resumeId}/extract-relationships`,
    null,
    {
      params: { mode },
      timeout: 45000,
    }
  );
  return response.data;
}

/**
 * Retrieves previously extracted relationships for a resume.
 * GET /api/v1/relationships/{resume_id}?mode=advanced
 */
export async function getRelationships(resumeId, mode = 'advanced') {
  try {
    const response = await apiClient.get(`/relationships/${resumeId}`, {
      params: { mode },
      timeout: 15000,
    });
    return response.data;
  } catch (error) {
    // Backend returns 404 if no relationships have been generated yet
    if (error.status === 404) {
      return {
        resume_id: resumeId,
        extraction_mode: mode,
        total_relationships: 0,
        relationships: [],
      };
    }
    throw error;
  }
}

/**
 * Triggers RAG indexing, chunk embedding, and candidate claim extraction.
 * POST /api/v1/rag/{resume_id}/index
 */
export async function indexResume(resumeId) {
  const response = await apiClient.post(`/rag/${resumeId}/index`, null, {
    timeout: 60000,
  });
  return response.data;
}

/**
 * Retrieves all extracted atomic claims for a candidate.
 * GET /api/v1/rag/claims/{resume_id}
 */
export async function getClaims(resumeId) {
  const response = await apiClient.get(`/rag/claims/${resumeId}`, {
    timeout: 15000,
  });
  return response.data;
}
