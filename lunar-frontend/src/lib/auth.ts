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
 *
 * Step 13: NO offline fallback. A forged local token would bypass every
 * Step 12 backend control (auth on /register, /refresh, ingest upload).
 * Network failures are re-thrown so the UI shows the error banner.
 */
export async function register(
  name: string,
  email: string,
  password: string
): Promise<TokenResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
  } catch {
    throw new Error(
      `Could not reach the auth service at ${API_BASE}. Check that the backend is running.`
    );
  }

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
 *
 * Step 13: NO offline fallback and NO demo bypass (see register()). A
 * backend-issued token is the only credential this app accepts.
 */
export async function login(
  email: string,
  password: string
): Promise<TokenResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
  } catch {
    throw new Error(
      `Could not reach the auth service at ${API_BASE}. Check that the backend is running.`
    );
  }

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
 * Decode the JWT `exp` claim without verifying (signature is verified
 * server-side on every request). Returns the expiry epoch seconds, or null
 * when the token is malformed. Short-term client-side guard so an expired
 * token never renders the app as authenticated; the backend 401 remains
 * authoritative. Long-term: move to an httpOnly session cookie so JS can
 * never read the token at all (needs backend Set-Cookie support).
 */
export function decodeTokenExp(token: string): number | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const payload = JSON.parse(
      atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"))
    ) as { exp?: unknown };
    return typeof payload.exp === "number" && Number.isFinite(payload.exp)
      ? payload.exp
      : null;
  } catch {
    return null;
  }
}

/**
 * True when the token is missing, malformed, or past its `exp` claim
 * (60s clock-skew grace). Malformed tokens are treated as expired so they
 * can never authenticate the UI.
 */
export function isTokenExpired(token: string | null): boolean {
  if (!token) return true;
  const exp = decodeTokenExp(token);
  if (exp === null) return true;
  return exp * 1000 <= Date.now() + 60_000;
}

/**
 * Check if the user currently has a stored, unexpired auth token.
 * An expired (or malformed) token is cleared so the login wall appears
 * instead of a cascade of 401 banners. The backend 401 remains the
 * authoritative gate — this is a UI fast-path only.
 */
export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  if (isTokenExpired(token)) {
    logout();
    return false;
  }
  return true;
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
 * Returns an empty object if not authenticated. A client-side expired
 * token is dropped (and cleared) rather than sent to fail as a 401.
 */
export function getAuthHeaders(): Record<string, string> {
  const token = getToken();
  if (!token || isTokenExpired(token)) {
    if (token) logout();
    return {};
  }
  return { Authorization: `Bearer ${token}` };
}
