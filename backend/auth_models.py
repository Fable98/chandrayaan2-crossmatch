"""
auth_models.py — Pydantic schemas for the authentication system.

Defines request/response models for user registration, login, and
token-based session management.
"""

from pydantic import BaseModel, Field, EmailStr


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """Payload for POST /auth/register."""
    name: str = Field(..., min_length=1, max_length=100, description="Display name")
    email: str = Field(..., min_length=3, max_length=255, description="Email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")


class UserLogin(BaseModel):
    """Payload for POST /auth/login."""
    email: str = Field(..., description="Registered email address")
    password: str = Field(..., description="Account password")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UserResponse(BaseModel):
    """Public user profile (never includes password hash)."""
    id: str
    name: str
    email: str
    created_at: str


class TokenResponse(BaseModel):
    """JWT token returned after successful login."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
