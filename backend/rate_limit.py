"""
rate_limit.py — Step 12: rate limiting + brute-force lockout for /auth/*.

Two layers:
  1. SlowAPI per-IP rate limits on auth routes (memory-backed by default,
     Redis-backed when REDIS_URL is configured and reachable).
  2. Account lockout: N failed logins for one email within a window locks
     that email for AUTH_LOCKOUT_MINUTES (tracked in Redis when available,
     else process memory). All authentication failures cost identical work
     and return identical messages so accounts cannot be enumerated.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("backend.rate_limit")

_limiter = None
_redis_client = None
_lockout_memory: dict[str, dict] = {}


def get_redis():
    """Shared Redis client or None (import-safe, connection-safe)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import os

        url = os.environ.get("REDIS_URL")
        if not url:
            try:
                from config import settings  # type: ignore[import-not-found]

                url = getattr(settings, "REDIS_URL", None)
            except Exception:
                url = None
        if not url:
            return None
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("Rate-limit backend: Redis active.")
        return client
    except Exception as exc:
        logger.warning("Rate-limit backend: Redis unavailable (%s); using memory.", exc)
        return None


def get_limiter():
    """Process-wide SlowAPI limiter (creates on first use)."""
    global _limiter
    if _limiter is not None:
        return _limiter
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    _limiter = Limiter(key_func=get_remote_address)
    return _limiter


def _lockout_conf() -> tuple[int, int]:
    try:
        from config import settings  # type: ignore[import-not-found]

        return int(getattr(settings, "AUTH_LOCKOUT_ATTEMPTS", 5)), int(
            getattr(settings, "AUTH_LOCKOUT_MINUTES", 15)
        )
    except Exception:
        return 5, 15


def _lockout_key(email: str) -> str:
    return f"auth_lockout:{email.lower().strip()}"


def is_locked_out(email: str) -> tuple[bool, int]:
    """(locked, retry_after_seconds) for this email. Never raises."""
    try:
        max_attempts, lock_minutes = _lockout_conf()
        now = time.time()
        client = get_redis()
        if client is not None:
            try:
                raw = client.get(_lockout_key(email))
                if not raw:
                    return False, 0
                import json

                data = json.loads(raw)
                fails = [t for t in data.get("fails", []) if now - t < lock_minutes * 60]
                if len(fails) >= max_attempts:
                    retry = int(max(fails) + lock_minutes * 60 - now)
                    return True, max(retry, 1)
                return False, 0
            except Exception:
                pass
        data = _lockout_memory.get(email.lower().strip())
        if not data:
            return False, 0
        fails = [t for t in data.get("fails", []) if now - t < lock_minutes * 60]
        if len(fails) >= max_attempts:
            retry = int(max(fails) + lock_minutes * 60 - now)
            return True, max(retry, 1)
        return False, 0
    except Exception:
        return False, 0


def record_failed_login(email: str) -> None:
    """Record one failed login. Never raises."""
    try:
        max_attempts, lock_minutes = _lockout_conf()
        now = time.time()
        key = email.lower().strip()
        client = get_redis()
        if client is not None:
            try:
                import json

                raw = client.get(_lockout_key(email))
                data = json.loads(raw) if raw else {"fails": []}
                fails = [t for t in data.get("fails", []) if now - t < lock_minutes * 60]
                fails.append(now)
                client.setex(_lockout_key(email), lock_minutes * 60,
                             json.dumps({"fails": fails}))
                return
            except Exception:
                pass
        data = _lockout_memory.setdefault(key, {"fails": []})
        data["fails"] = [t for t in data.get("fails", []) if now - t < lock_minutes * 60]
        data["fails"].append(now)
    except Exception:
        pass


def record_successful_login(email: str) -> None:
    """Clear failure history on success. Never raises."""
    try:
        key = email.lower().strip()
        _lockout_memory.pop(key, None)
        client = get_redis()
        if client is not None:
            try:
                client.delete(_lockout_key(email))
            except Exception:
                pass
    except Exception:
        pass


def _reset_for_tests() -> None:
    """Clear in-memory lockout state (tests)."""
    _lockout_memory.clear()
