/**
 * auth.ts — Frontend authentication utilities for the Chandrayaan-2 Crossmatch app.
 *
 * Provides login, register, logout, and session management using JWT tokens
 * stored in localStorage. Communicates with the FastAPI /auth/* endpoints.
 */

import { API_BASE } from "./api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

const TOKEN_KEY = "astralynx_auth_token";
const USER_KEY = "astralynx_auth_user";

function setToken(token: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(TOKEN_KEY, token);
  }
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

function setStoredUser(user: AuthUser): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }
}

function getStoredUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Register a new user account.
 * On success, the JWT token is stored and the user is "logged in".
 */
export async function register(
  name: string,
  email: string,
  password: string
): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }));
    throw new Error(err.detail || "Registration failed");
  }

  const data: TokenResponse = await res.json();
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

/**
 * Log in with email and password.
 * On success, the JWT token is stored locally.
 */
export async function login(
  email: string,
  password: string
): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Invalid email or password");
  }

  const data: TokenResponse = await res.json();
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

/**
 * Log out — clears the stored JWT token and user data.
 */
export function logout(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }
}

/**
 * Check if the user currently has a stored auth token.
 */
export function isAuthenticated(): boolean {
  return !!getToken();
}

/**
 * Get the currently stored user (from localStorage, no network call).
 */
export function getCurrentUser(): AuthUser | null {
  return getStoredUser();
}

/**
 * Fetch the current user profile from the backend (validates the token).
 * Returns null if the token is invalid or expired.
 */
export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const token = getToken();
  if (!token) return null;

  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      // Token is invalid/expired — clear it
      logout();
      return null;
    }
    const user: AuthUser = await res.json();
    setStoredUser(user);
    return user;
  } catch {
    return null;
  }
}

/**
 * Get the Authorization header value for authenticated API requests.
 * Returns an empty object if not authenticated.
 */
export function getAuthHeaders(): Record<string, string> {
  const token = getToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}
