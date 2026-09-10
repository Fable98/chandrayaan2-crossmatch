"""
routers/auth.py — Authentication router for the SIH26166 Lunar backend.

JWT-based user registration, login, and session validation with Step 12
hardening: no-default JWT secret (fail closed), bcrypt cost 12, per-IP
rate limits plus per-account lockout on /auth/*, and users in
Postgres/Supabase when a database URL is configured (JSON file otherwise).
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt

from config import settings
import auth_store
from auth_models import UserCreate, UserLogin, UserResponse, TokenResponse
from rate_limit import get_limiter, is_locked_out, record_failed_login, record_successful_login


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["authentication"])
limiter = get_limiter()


def _bcrypt_rounds() -> int:
    try:
        return int(getattr(settings, "BCRYPT_ROUNDS", 12))
    except Exception:
        return 12


# Password hashing helpers (using bcrypt directly to avoid passlib compat issues)
def _hash_password(password: str) -> str:
    """Hash a password using bcrypt (cost factor 12)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=_bcrypt_rounds())).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash (constant-time compare)."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _jwt_secret() -> str:
    """Fail closed when no secret is configured (no default, no fallback)."""
    return settings.require_jwt_secret()


# JWT bearer scheme for dependency injection
bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_access_token(user_id: str) -> str:
    """Create a signed JWT access token for the given user ID."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRY_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=settings.JWT_ALGORITHM)


def _decode_token(token: str) -> Optional[str]:
    """Decode and validate a JWT token, returning the user ID or None."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except (JWTError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency that extracts and validates the current user
    from the Authorization: Bearer <token> header.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = _decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_store.find_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def _user_to_response(user: dict) -> UserResponse:
    """Convert an internal user dict to the public UserResponse model."""
    return UserResponse(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        created_at=user["created_at"],
    )


def _auth_limit() -> str:
    """Dynamic SlowAPI limit: re-read per request so tests/ops can retune
    AUTH_RATE_LIMIT without reimporting (zero-arg callable contract)."""
    try:
        return str(getattr(settings, "AUTH_RATE_LIMIT", "10/minute"))
    except Exception:
        return "10/minute"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(_auth_limit)
async def register(request: Request, body: UserCreate):
    """
    Register a new user account.

    Returns a JWT access token so the user is automatically logged in
    after registration.
    """
    # Check if email is already taken
    if auth_store.find_user_by_email(body.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create user record
    user = {
        "id": str(uuid.uuid4()),
        "name": body.name.strip(),
        "email": body.email.lower().strip(),
        "password_hash": _hash_password(body.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Persist (Postgres when configured, else the JSON flat file)
    auth_store.create_user(user)

    # Generate token and respond
    token = _create_access_token(user["id"])
    return TokenResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(_auth_limit)
async def login(request: Request, body: UserLogin):
    """
    Authenticate with email and password.

    Returns a JWT access token on success. Failures are indistinguishable
    (no account enumeration) and count toward temporary account lockout.
    """
    locked, retry_after = is_locked_out(body.email)
    if locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = auth_store.find_user_by_email(body.email)

    if user is None or not _verify_password(body.password, user["password_hash"]):
        record_failed_login(body.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    record_successful_login(body.email)
    token = _create_access_token(user["id"])
    return TokenResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the profile of the currently authenticated user.

    Requires a valid Bearer token in the Authorization header.
    """
    return _user_to_response(current_user)
