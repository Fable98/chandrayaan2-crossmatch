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
JOB_TTL_SECONDS = 7 * 24 * 3600


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

    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, socket_connect_timeout=2,
                                           socket_timeout=2, decode_responses=True)
        self.client.ping()

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _logs_key(self, job_id: str) -> str:
        return f"job:{job_id}:logs"

    def _load(self, job_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.hgetall(self._key(job_id))
        if not raw:
            return None
        job = {k: (json.loads(v) if k in ("result",) else v) for k, v in raw.items()}
        try:
            job["progress"] = float(job.get("progress", 0.0))
        except Exception:
            job["progress"] = 0.0
        logs = self.client.lrange(self._logs_key(job_id), 0, -1)
        job["logs"] = logs
        return job

    def create(self, job_id: str, job: Dict[str, Any]) -> None:
        payload = {k: (json.dumps(v) if k == "result" else str(v) if v is not None else "")
                   for k, v in job.items() if k != "logs"}
        self.client.hset(self._key(job_id), mapping=payload)
        self.client.delete(self._logs_key(job_id))
        for line in job.get("logs", [])[-LOG_CAP:]:
            self.client.rpush(self._logs_key(job_id), line)
        self.client.expire(self._key(job_id), JOB_TTL_SECONDS)
        self.client.expire(self._logs_key(job_id), JOB_TTL_SECONDS)

    def update(self, job_id: str, fields: Dict[str, Any]) -> None:
        fields = {k: v for k, v in fields.items() if k != "logs"}
        if not fields:
            return
        payload = {k: (json.dumps(v) if k == "result" else str(v) if v is not None else "")
                   for k, v in fields.items()}
        self.client.hset(self._key(job_id), mapping=payload)
        self.client.expire(self._key(job_id), JOB_TTL_SECONDS)

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
            self.client.expire(self._logs_key(job_id), JOB_TTL_SECONDS)
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

class DbJobStore:
    """SQL table job store (Postgres via DATABASE_URL, sqlite for tests)."""

    def __init__(self, url: str) -> None:
        from sqlalchemy import JSON, Column, DateTime, Float, String, Text, create_engine, func
        from sqlalchemy.orm import declarative_base, sessionmaker

        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        Base = declarative_base()

        class Job(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "jobs"
            id = Column(String, primary_key=True)
            type = Column(String, default="")
            status = Column(String, default="")
            progress = Column(Float, default=0.0)
            current_phase = Column(String, default="")
            result = Column(JSON, nullable=True)
            error = Column(Text, nullable=True)
            logs = Column(JSON, default=list)
            updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                                onupdate=func.now())

        self._Job = Job
        Base.metadata.create_all(self._engine)
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
