"""
job_store.py — Step 12: pluggable JobManager storage.

Selection order (first configured-and-reachable wins):
  1. Redis (REDIS_URL) — queue-friendly result backend with TTL.
  2. SQLAlchemy table (USERS_DATABASE_URL / DATABASE_URL, incl. sqlite for
     tests) — genuinely DB-backed job rows.
  3. Process memory (local dev / single-server fallback).

The in-memory behavior (bounded per-job log ring, snapshot get_logs) is
identical across backends so routers need no changes to switch.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("backend.job_store")

LOG_CAP = 200


def _job_ttl_seconds() -> int:
    """Job-record TTL in seconds from settings (JOB_TTL_HOURS, default 7d).

    Centralized in backend/config.py with env override (Phase 6); the literal
    below is the last-resort fallback when settings are unimportable.
    """
    try:
        from config import settings  # type: ignore[import-not-found]

        return int(getattr(settings, "JOB_TTL_HOURS", 168)) * 3600
    except Exception:
        return 7 * 24 * 3600


JOB_TTL_SECONDS = _job_ttl_seconds()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# 3. Process memory (fallback)
# ---------------------------------------------------------------------------

class MemoryJobStore:
    """Thread-safe in-memory job store (single-server fallback)."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, job: Dict[str, Any]) -> None:
        with self._lock:
            self.jobs[job_id] = job

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(fields)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def append_log(self, job_id: str, line: str, cap: int = LOG_CAP) -> None:
        try:
            with self._lock:
                job = self.jobs.get(job_id)
                if job is None:
                    return
                logs = job.setdefault("logs", [])
                logs.append(f"[{_now()}] {line}")
                if len(logs) > cap:
                    del logs[: len(logs) - cap]
        except Exception:
            pass

    def get_logs(self, job_id: str, after: int = 0) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            logs = list(job.get("logs", []))
        after = max(0, int(after))
        return {"job_id": job_id, "total": len(logs), "after": after,
                "lines": logs[after:], "status": job.get("status")}


# ---------------------------------------------------------------------------
# 1. Redis backend
# ---------------------------------------------------------------------------

class RedisJobStore:
    """Redis-hash job store with TTL. Raises on construction if unreachable."""

    # Structured fields are JSON-encoded on write and JSON-decoded on read
    # (previously only "result" round-tripped; "error" dicts degraded to
    # str(dict) and came back as opaque strings).
    _JSON_FIELDS = frozenset({"result", "error"})

    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, socket_connect_timeout=2,
                                           socket_timeout=2, decode_responses=True)
        self.client.ping()

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _logs_key(self, job_id: str) -> str:
        return f"job:{job_id}:logs"

    @classmethod
    def _encode_value(cls, key: str, value: Any) -> str:
        if value is None:
            return ""
        if key in cls._JSON_FIELDS:
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    @classmethod
    def _decode_value(cls, key: str, raw: str) -> Any:
        if key in cls._JSON_FIELDS:
            if raw == "":
                return None
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return raw
        return raw

    def _load(self, job_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.hgetall(self._key(job_id))
        if not raw:
            return None
        job = {k: self._decode_value(k, v) for k, v in raw.items()}
        try:
            job["progress"] = float(job.get("progress", 0.0))
        except Exception:
            job["progress"] = 0.0
        logs = self.client.lrange(self._logs_key(job_id), 0, -1)
        job["logs"] = logs
        return job

    def create(self, job_id: str, job: Dict[str, Any]) -> None:
        payload = {k: self._encode_value(k, v)
                   for k, v in job.items() if k != "logs"}
        self.client.hset(self._key(job_id), mapping=payload)
        self.client.delete(self._logs_key(job_id))
        for line in job.get("logs", [])[-LOG_CAP:]:
            self.client.rpush(self._logs_key(job_id), line)
        self.client.expire(self._key(job_id), _job_ttl_seconds())
        self.client.expire(self._logs_key(job_id), _job_ttl_seconds())

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        fields = {k: v for k, v in fields.items() if k != "logs"}
        if not fields:
            return
        payload = {k: self._encode_value(k, v)
                   for k, v in fields.items()}
        self.client.hset(self._key(job_id), mapping=payload)
        self.client.expire(self._key(job_id), _job_ttl_seconds())

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._load(job_id)
        except Exception as exc:
            logger.warning("Redis job fetch failed (%s).", exc)
            return None

    def append_log(self, job_id: str, line: str, cap: int = LOG_CAP) -> None:
        try:
            self.client.rpush(self._logs_key(job_id), f"[{_now()}] {line}")
            self.client.ltrim(self._logs_key(job_id), -cap, -1)
            self.client.expire(self._logs_key(job_id), _job_ttl_seconds())
        except Exception:
            pass

    def get_logs(self, job_id: str, after: int = 0) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        if job is None:
            return None
        logs = job.get("logs", [])
        after = max(0, int(after))
        return {"job_id": job_id, "total": len(logs), "after": after,
                "lines": logs[after:], "status": job.get("status")}


# ---------------------------------------------------------------------------
# 2. SQLAlchemy DB backend
# ---------------------------------------------------------------------------

# Module-level Base/Job: defining the declarative model inside __init__ rebuilt
# the "jobs" table on every DbJobStore() (SAWarning: table already defined)
# and leaked duplicate metadata. One definition, reused by all instances.
try:
    from sqlalchemy import JSON as _SA_JSON
    from sqlalchemy import Column as _SA_Column
    from sqlalchemy import DateTime as _SA_DateTime
    from sqlalchemy import Float as _SA_Float
    from sqlalchemy import String as _SA_String
    from sqlalchemy import Text as _SA_Text
    from sqlalchemy import func as _SA_func
    from sqlalchemy.orm import declarative_base as _sa_declarative_base

    _JobsBase = _sa_declarative_base()

    class JobRow(_JobsBase):  # type: ignore[valid-type,misc]
        __tablename__ = "jobs"
        id = _SA_Column(_SA_String, primary_key=True)
        type = _SA_Column(_SA_String, default="")
        status = _SA_Column(_SA_String, default="")
        progress = _SA_Column(_SA_Float, default=0.0)
        current_phase = _SA_Column(_SA_String, default="")
        result = _SA_Column(_SA_JSON, nullable=True)
        error = _SA_Column(_SA_Text, nullable=True)
        logs = _SA_Column(_SA_JSON, default=list)
        updated_at = _SA_Column(_SA_DateTime(timezone=True), server_default=_SA_func.now(),
                                onupdate=_SA_func.now())
except Exception:  # pragma: no cover - SQLAlchemy missing
    _JobsBase = None  # type: ignore[assignment]
    JobRow = None  # type: ignore[assignment]


class DbJobStore:
    """SQL table job store (Postgres via DATABASE_URL, sqlite for tests)."""

    def __init__(self, url: str) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        if _JobsBase is None or JobRow is None:
            raise RuntimeError("SQLAlchemy is required for DbJobStore")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        self._Job = JobRow
        _JobsBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._lock = threading.Lock()

    def create(self, job_id: str, job: Dict[str, Any]) -> None:
        with self._lock, self._Session() as session:
            session.merge(self._Job(
                id=job_id, type=str(job.get("type", "")),
                status=str(job.get("status", "")), progress=float(job.get("progress", 0.0)),
                current_phase=str(job.get("current_phase", "")),
                result=job.get("result"), error=job.get("error"),
                logs=list(job.get("logs", [])[-LOG_CAP:]),
            ))
            session.commit()

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        fields = {k: v for k, v in fields.items() if k != "logs"}
        if not fields:
            return
        with self._lock, self._Session() as session:
            row = session.query(self._Job).filter(self._Job.id == job_id).first()
            if row is None:
                return
            for k, v in fields.items():
                if k == "progress":
                    try:
                        v = float(v)
                    except Exception:
                        continue
                if hasattr(row, k):
                    setattr(row, k, v)
            session.commit()

    def _row_to_job(self, row) -> Dict[str, Any]:
        return {"status": row.status, "type": row.type, "progress": float(row.progress or 0.0),
                "current_phase": row.current_phase or "", "result": row.result,
                "error": row.error, "logs": list(row.logs or [])}

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._Session() as session:
                row = session.query(self._Job).filter(self._Job.id == job_id).first()
                return self._row_to_job(row) if row else None
        except Exception as exc:
            logger.warning("DB job fetch failed (%s).", exc)
            return None

    def append_log(self, job_id: str, line: str, cap: int = LOG_CAP) -> None:
        try:
            with self._lock, self._Session() as session:
                row = session.query(self._Job).filter(self._Job.id == job_id).first()
                if row is None:
                    return
                logs = list(row.logs or [])
                logs.append(f"[{_now()}] {line}")
                row.logs = logs[-cap:]
                session.commit()
        except Exception:
            pass

    def get_logs(self, job_id: str, after: int = 0) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        if job is None:
            return None
        logs = job.get("logs", [])
        after = max(0, int(after))
        return {"job_id": job_id, "total": len(logs), "after": after,
                "lines": logs[after:], "status": job.get("status")}


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def build_job_store():
    """Redis -> DB -> memory. Logs which backend won (no silent downgrade)."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        try:
            from config import settings  # type: ignore[import-not-found]

            redis_url = getattr(settings, "REDIS_URL", None)
        except Exception:
            redis_url = None
    if redis_url:
        try:
            store = RedisJobStore(redis_url)
            logger.info("Job store: Redis backend active.")
            return store
        except Exception as exc:
            logger.warning("Job store: Redis unreachable (%s); trying DB.", exc)
    db_url = os.environ.get("USERS_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        try:
            from config import settings  # type: ignore[import-not-found]

            db_url = getattr(settings, "USERS_DATABASE_URL", None) or getattr(
                settings, "DATABASE_URL", None)
        except Exception:
            db_url = None
    if db_url:
        try:
            store = DbJobStore(db_url)
            logger.info("Job store: DB backend active.")
            return store
        except Exception as exc:
            logger.warning("Job store: DB unreachable (%s); using memory.", exc)
    logger.info("Job store: in-memory backend (single-server fallback).")
    return MemoryJobStore()
