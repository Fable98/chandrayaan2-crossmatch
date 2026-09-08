"""
database.py — Optional Supabase PostgreSQL connection helper.

The primary data path for the SIH26166 backend is flat-file JSON manifests
and tile assets. This module provides a clean, lazily-initialized database
interface when DATABASE_URL is supplied in the environment (e.g. for user
sessions, logging, or database-backed triplets), without rewriting the
existing flat-file/JSON data serving pipeline.
"""

from typing import Optional
from config import settings


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
