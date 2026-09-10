"""
auth_store.py — Step 12: user account storage.

Primary: Postgres/Supabase via SQLAlchemy when a database URL is configured
(USERS_DATABASE_URL, else DATABASE_URL). Fallback: the legacy flat JSON
file (local dev / tests without a database). The rest of the auth code talks
only to this module, so moving users to Postgres requires no router changes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("backend.auth_store")

BACKEND_DIR = Path(__file__).resolve().parent
USERS_FILE = BACKEND_DIR / "data" / "users.json"


def _database_url() -> Optional[str]:
    """Best-effort database URL without importing config at module load."""
    url = os.environ.get("USERS_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        from config import settings  # type: ignore[import-not-found]

        url = getattr(settings, "USERS_DATABASE_URL", None) or getattr(settings, "DATABASE_URL", None)
        return url
    except Exception:
        return None


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


# ---------------------------------------------------------------------------
# SQLAlchemy backend (optional)
# ---------------------------------------------------------------------------

_engine = None
_engine_failed = False
_SessionLocal = None
_UserModel = None


def _ensure_engine():
    """Lazily build the SQLAlchemy engine + users table. Returns None w/o DB."""
    global _engine, _engine_failed, _SessionLocal, _UserModel
    if _engine is not None or _engine_failed:
        return _engine
    url = _database_url()
    if not url:
        return None
    try:
        from sqlalchemy import Column, DateTime, String, create_engine, func
        from sqlalchemy.orm import declarative_base, sessionmaker

        url = _normalize_url(url)
        connect_args: dict = {}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        Base = declarative_base()

        class User(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "users"
            id = Column(String, primary_key=True)
            name = Column(String, nullable=False)
            email = Column(String, nullable=False, unique=True, index=True)
            password_hash = Column(String, nullable=False)
            created_at = Column(DateTime(timezone=True), server_default=func.now())

        _UserModel = User
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
        logger.info("User store: Postgres/SQLAlchemy backend active.")
        return _engine
    except Exception as exc:
        logger.warning("User store: database backend unavailable (%s); using JSON file.", exc)
        _engine_failed = True  # negative cache
        return None


def _reset_for_tests() -> None:
    """Drop cached engine state (tests that switch database URLs)."""
    global _engine, _engine_failed, _SessionLocal, _UserModel
    try:
        if _engine is not None:
            _engine.dispose()
    except Exception:
        pass
    _engine = None
    _engine_failed = False
    _SessionLocal = None
    _UserModel = None


def using_database() -> bool:
    """True when the Postgres/SQLAlchemy user store is active."""
    try:
        return _ensure_engine() is not None
    except Exception:
        return False


def _row_to_dict(row) -> dict:
    created = row.created_at.isoformat() if getattr(row, "created_at", None) else ""
    return {
        "id": row.id,
        "name": row.name,
        "email": row.email,
        "password_hash": row.password_hash,
        "created_at": created,
    }


# ---------------------------------------------------------------------------
# JSON flat-file backend (fallback)
# ---------------------------------------------------------------------------

def _load_users_json() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    try:
        with open(USERS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_users_json(users: list[dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email (case-insensitive). Never raises."""
    try:
        if _ensure_engine():
            assert _SessionLocal is not None and _UserModel is not None
            with _SessionLocal() as session:
                row = (
                    session.query(_UserModel)
                    .filter(_UserModel.email == email.lower().strip())
                    .first()
                )
                return _row_to_dict(row) if row else None
    except Exception as exc:
        logger.warning("DB user lookup failed (%s); falling back to JSON.", exc)
    for user in _load_users_json():
        if user.get("email", "").lower() == email.lower().strip():
            return user
    return None


def find_user_by_id(user_id: str) -> Optional[dict]:
    """Look up a user by ID. Never raises."""
    try:
        if _ensure_engine():
            assert _SessionLocal is not None and _UserModel is not None
            with _SessionLocal() as session:
                row = session.query(_UserModel).filter(_UserModel.id == user_id).first()
                return _row_to_dict(row) if row else None
    except Exception as exc:
        logger.warning("DB user lookup failed (%s); falling back to JSON.", exc)
    for user in _load_users_json():
        if user.get("id") == user_id:
            return user
    return None


def create_user(user: dict) -> dict:
    """Persist a new user record (DB when configured, else JSON file)."""
    try:
        if _ensure_engine():
            assert _SessionLocal is not None and _UserModel is not None
            with _SessionLocal() as session:
                row = _UserModel(
                    id=user["id"],
                    name=user["name"],
                    email=user["email"].lower().strip(),
                    password_hash=user["password_hash"],
                )
                session.add(row)
                session.commit()
                return find_user_by_id(user["id"]) or user
    except Exception as exc:
        logger.warning("DB user insert failed (%s); falling back to JSON.", exc)
    users = _load_users_json()
    users.append(user)
    _save_users_json(users)
    return user
