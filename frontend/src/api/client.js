import axios from 'axios';

// Resolve API Base URL from environment or sensible development fallback
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const TOKEN_STORAGE_KEY = 'intervai_access_token';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
});

// Request interceptor: Attach bearer token if present in sessionStorage
apiClient.interceptors.request.use(
  (config) => {
    // Only attach authorization to requests matching our base API URL
    const isRelativeOrMatchesBase =
      !config.url.startsWith('http://') && !config.url.startsWith('https://') ||
      config.url.startsWith(API_BASE_URL);

    if (isRelativeOrMatchesBase) {
      const token = sessionStorage.getItem(TOKEN_STORAGE_KEY);
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: Normalize backend errors (FastAPI detail strings/arrays)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let errorMessage = 'An unexpected error occurred. Please try again.';
    let errorCategory = 'UNKNOWN';

    // 1. HTTP Response received from FastAPI backend
    if (error.response) {
      const { status, data } = error.response;

      if (data && data.detail) {
        if (typeof data.detail === 'string') {
          errorMessage = data.detail;
        } else if (Array.isArray(data.detail)) {
          // FastAPI 422 validation errors: array of { loc, msg, type }
          errorMessage = data.detail.map((err) => err.msg || JSON.stringify(err)).join('; ');
        }
      } else if (data && data.message) {
        errorMessage = data.message;
      } else if (status === 400) {
        errorMessage = 'Invalid request parameters. Please verify your input and try again.';
        errorCategory = 'BAD_REQUEST';
      } else if (status === 401) {
        errorMessage = 'Your session has expired or you are unauthorized. Please sign in again.';
        errorCategory = 'UNAUTHORIZED';
      } else if (status === 403) {
        errorMessage = 'You do not have permission to access this interview or resume.';
        errorCategory = 'FORBIDDEN';
      } else if (status === 404) {
        errorMessage = 'The requested resource was not found.';
        errorCategory = 'NOT_FOUND';
      } else if (status === 409) {
        errorMessage = data?.detail || 'An operation is already in progress for this interview session. Please wait a moment.';
        errorCategory = 'CONFLICT';
      } else if (status === 413) {
        errorMessage = 'The uploaded document exceeds the maximum permitted file size.';
        errorCategory = 'FILE_TOO_LARGE';
      } else if (status === 415) {
        errorMessage = 'Unsupported file format. Please upload a valid PDF document.';
        errorCategory = 'UNSUPPORTED_MEDIA';
      } else if (status === 429) {
        errorMessage = 'The AI provider is experiencing high traffic or rate limits. Please wait a few seconds and retry.';
        errorCategory = 'RATE_LIMITED';
      } else if (status === 502 || status === 503 || status === 504) {
        errorMessage = 'The AI service is temporarily unavailable or experiencing high demand. Please try again in a few moments.';
        errorCategory = 'GATEWAY_OR_PROVIDER_UNAVAILABLE';
      } else if (status >= 500) {
        errorMessage = data?.detail || 'An internal server error occurred. Please try again later.';
        errorCategory = 'INTERNAL_SERVER_ERROR';
      }
    }
    // 2. Request was made but no response received (Timeout, Network disconnect, CORS)
    else if (error.code === 'ECONNABORTED' || (error.message && error.message.toLowerCase().includes('timeout'))) {
      errorMessage = 'The request timed out while waiting for AI generation. The server may still be processing; you can safely retry.';
      errorCategory = 'TIMEOUT';
    } else if (typeof navigator !== 'undefined' && !navigator.onLine) {
      errorMessage = 'You appear to be offline. Please check your internet connection and try again.';
      errorCategory = 'OFFLINE';
    } else if (error.code === 'ERR_NETWORK' || error.request) {
      errorMessage = 'Unable to reach the IntervAI server. Please verify your connection or ensure the backend server is running.';
      errorCategory = 'NETWORK_ERROR';
    }

    const enhancedError = new Error(errorMessage);
    enhancedError.status = error.response ? error.response.status : null;
    enhancedError.category = errorCategory;
    enhancedError.data = error.response ? error.response.data : null;
    enhancedError.code = error.code;
    enhancedError.originalError = error;

    return Promise.reject(enhancedError);
  }
);

export default apiClient;
