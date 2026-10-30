import apiClient from './client';

/**
 * Registers a new user account with full name, email, and password.
 * Backend responds with HTTP 201 and UserResponse { id, email, full_name, created_at }.
 */
export async function registerUser({ fullName, email, password }) {
  const payload = {
    full_name: fullName.trim(),
    email: email.trim().toLowerCase(),
    password,
  };
  const response = await apiClient.post('/auth/register', payload);
  return response.data;
}

/**
 * Authenticates user credentials and obtains a signed JWT access token.
 * Backend responds with HTTP 200 and TokenResponse { access_token, token_type, user }.
 */
export async function loginUser({ email, password }) {
  const payload = {
    email: email.trim().toLowerCase(),
    password,
  };
  const response = await apiClient.post('/auth/login', payload);
  return response.data;
}

/**
 * Retrieves profile of the currently authenticated user using stored Bearer token.
 * Backend responds with HTTP 200 and UserResponse { id, email, full_name, created_at }.
 */
export async function getCurrentUser() {
  const response = await apiClient.get('/auth/me');
  return response.data;
}
