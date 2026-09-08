"""
routers/auth.py — Authentication router for the SIH26166 Lunar backend.

Provides JWT-based user registration, login, and session validation.
User data is stored in a flat JSON file (data/users.json), consistent
with the project's existing flat-file data serving pattern.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt

from config import settings
from auth_models import UserCreate, UserLogin, UserResponse, TokenResponse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["authentication"])

# Password hashing helpers (using bcrypt directly to avoid passlib compat issues)
def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

# JWT bearer scheme for dependency injection
bearer_scheme = HTTPBearer(auto_error=False)

# Flat-file user store path
BACKEND_DIR = Path(__file__).resolve().parent.parent
USERS_FILE = BACKEND_DIR / "data" / "users.json"


# ---------------------------------------------------------------------------
# User store — flat-file JSON helpers
# ---------------------------------------------------------------------------

def _load_users() -> list[dict]:
    """Load all users from the flat-file JSON store."""
    if not USERS_FILE.exists():
        return []
    try:
        with open(USERS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_users(users: list[dict]) -> None:
    """Persist users list to the flat-file JSON store."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _find_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email (case-insensitive)."""
    users = _load_users()
    email_lower = email.lower().strip()
    for user in users:
        if user.get("email", "").lower() == email_lower:
            return user
    return None


def _find_user_by_id(user_id: str) -> Optional[dict]:
    """Look up a user by ID."""
    users = _load_users()
    for user in users:
        if user.get("id") == user_id:
            return user
    return None


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
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _decode_token(token: str) -> Optional[str]:
    """Decode and validate a JWT token, returning the user ID or None."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
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

    user = _find_user_by_id(user_id)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate):
    """
    Register a new user account.

    Returns a JWT access token so the user is automatically logged in
    after registration.
    """
    # Check if email is already taken
    if _find_user_by_email(body.email):
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

    # Persist
    users = _load_users()
    users.append(user)
    _save_users(users)

    # Generate token and respond
    token = _create_access_token(user["id"])
    return TokenResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """
    Authenticate with email and password.

    Returns a JWT access token on success.
    """
    user = _find_user_by_email(body.email)

    if user is None or not _verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

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
