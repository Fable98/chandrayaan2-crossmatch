"""
database.py — Optional Supabase PostgreSQL connection helper.

The primary data path for the SIH26166 backend is flat-file JSON manifests
and tile assets. This module provides a clean, lazily-initialized database
interface when DATABASE_URL is supplied in the environment (e.g. for user
sessions, logging, or database-backed triplets), without rewriting the
existing flat-file/JSON data serving pipeline.
"""

from __future__ import annotations

from typing import Iterator, Optional

from fastapi import HTTPException

try:
    from config import settings
except ImportError:  # pragma: no cover - direct-module test path
    from backend.config import settings  # type: ignore


def is_database_configured() -> bool:
    """Return True if DATABASE_URL is configured."""
    return bool(settings.DATABASE_URL)


def get_db_connection_string() -> Optional[str]:
    """
    Return the sanitized database connection URL, adjusting postgres://
    to postgresql:// if needed (Render/Supabase compatibility).
    """
    url = settings.DATABASE_URL
    if not url:
        return None
    # SQLAlchemy and psycopg require postgresql:// rather than legacy postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


# ---------------------------------------------------------------------------
# Real engine / session factory (previously a stub: connection string only).
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily build (and cache) the SQLAlchemy engine, or None when unconfigured."""
    global _engine
    if _engine is not None:
        return _engine
    url = get_db_connection_string()
    if not url:
        return None
    from sqlalchemy import create_engine

    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    return _engine


def get_session_factory():
    """Lazily build (and cache) the sessionmaker bound to get_engine()."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    engine = get_engine()
    if engine is None:
        return None
    from sqlalchemy.orm import sessionmaker

    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionLocal


def get_db() -> Iterator:
    """FastAPI dependency yielding a DB session (503 when unconfigured)."""
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured — database is unavailable.",
        )
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            session.close()
        except Exception:
            pass


def reset_engine_for_tests() -> None:
    """Dispose the cached engine (tests that swap DATABASE_URL)."""
    global _engine, _SessionLocal
    try:
        if _engine is not None:
            _engine.dispose()
    except Exception:
        pass
    _engine = None
    _SessionLocal = None
