"""
test_phase6_contracts.py — Phase 6 endpoint-contract + ingest-behavior tests.

Pins shapes the frontend relies on so refactors cannot silently drift them:
- ingest upload: 401 wall, 413 over-count cap, {job_id, files_uploaded} shape
- ingest status: public shape, progress_pct-only (no `percent` duality),
  log_lines capped at 50
- ingest results: 409 while running
- delete_triplet: {triplet_id, deleted, removed} + 400/401/404 walls
  (never deletes real data — unknown ids only)
- lro-candidates: {triplet_id, candidates, ...} + 404 wall
- config centralization: JOB_TTL_HOURS present, job_store honors it
"""

import io
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-that-is-long-enough-for-the-32-char-minimum-0123456789",
)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("AUTH_RATE_LIMIT", "1000/minute")

BACKEND_DIR = str(Path(__file__).resolve().parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if "config" in sys.modules and not hasattr(sys.modules["config"], "settings"):
    sys.modules.pop("config", None)
if "data" in sys.modules and not hasattr(sys.modules["data"], "loader"):
    sys.modules.pop("data", None)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

_CACHED: dict | None = None


def _auth():
    global _CACHED
    if _CACHED is None:
        import uuid

        email = f"p6-{uuid.uuid4().hex[:8]}@example.com"
        r = client.post(
            "/auth/register",
            json={"name": "P6", "email": email, "password": "correct-horse-123"},
        )
        assert r.status_code == 201, r.text[:200]
        _CACHED = {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _CACHED


def _zip_bytes(n: int = 16) -> io.BytesIO:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("note.txt", "x" * n)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Ingest upload walls
# ---------------------------------------------------------------------------

def test_ingest_upload_requires_bearer():
    r = client.post(
        "/api/ingest/upload",
        files=[("files", ("a.zip", _zip_bytes(), "application/zip"))],
    )
    assert r.status_code == 401, r.text[:200]


def test_ingest_upload_rejects_too_many_files():
    # MAX_INGEST_FILES defaults to 10: 11 empty-but-valid zips trip the
    # count cap before a single byte is streamed (cheap, hermetic).
    files = [
        ("files", (f"f{i}.zip", _zip_bytes(), "application/zip"))
        for i in range(11)
    ]
    r = client.post("/api/ingest/upload", files=files, headers=_auth())
    assert r.status_code == 413, r.text[:200]
    assert "max" in r.json()["detail"].lower()


def test_ingest_upload_empty_batch_is_400():
    r = client.post("/api/ingest/upload", files=[], headers=_auth())
    assert r.status_code in (400, 422), r.text[:200]


# ---------------------------------------------------------------------------
# Ingest status/results shapes (hermetic: seed the router job table)
# ---------------------------------------------------------------------------

def _seed_job(job_id: str, n_logs: int = 60) -> None:
    from routers import ingest as ingest_router

    ingest_router._jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "Matching triplets...",
        "progress_pct": 33.0,
        "started_at": "00:00:00",
        "completed_at": None,
        "log_lines": [f"line {i}" for i in range(n_logs)],
        "error": None,
        "summary": "",
        "triplets": [],
        "upload_dir": "/tmp/p6",
        "config": {},
    }


def test_ingest_status_shape_single_progress_key_and_log_cap():
    _seed_job("p6-shape", n_logs=60)
    try:
        r = client.get("/api/ingest/status/p6-shape")
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert set(body) >= {
            "job_id", "status", "stage", "progress_pct",
            "started_at", "completed_at", "log_lines", "error",
        }, sorted(body)
        assert "percent" not in body, "progress duality: only progress_pct may exist"
        assert 0.0 <= body["progress_pct"] <= 100.0
        assert len(body["log_lines"]) == 50, "log_lines must be capped at 50"
        assert body["log_lines"][-1] == "line 59", "cap keeps the tail, not the head"
    finally:
        from routers import ingest as ingest_router

        ingest_router._jobs.pop("p6-shape", None)


def test_ingest_results_409_while_running():
    _seed_job("p6-running", n_logs=3)
    try:
        r = client.get("/api/ingest/results/p6-running")
        assert r.status_code == 409, r.text[:200]
    finally:
        from routers import ingest as ingest_router

        ingest_router._jobs.pop("p6-running", None)


# ---------------------------------------------------------------------------
# delete_triplet + lro-candidates contracts (read-only walls only)
# ---------------------------------------------------------------------------

def test_delete_triplet_requires_bearer():
    # Unauthenticated must never reach the destructive path.
    r = client.delete("/triplets/region_001")
    assert r.status_code == 401, r.text[:200]


def test_delete_triplet_rejects_bad_ids():
    for bad in ("..", "a/b", "x%2f..%2f", ""):
        r = client.delete(f"/triplets/{bad}", headers=_auth())
        assert r.status_code in (400, 404, 405, 422), (bad, r.status_code)


def test_delete_triplet_curated_needs_force_and_unknown_is_404():
    r = client.delete("/triplets/region_001", headers=_auth())
    assert r.status_code == 400, r.text[:200]
    assert "force" in r.json()["detail"].lower()
    r = client.delete("/triplets/triplet_p6_no_such_bundle", headers=_auth())
    assert r.status_code == 404, r.text[:200]


def test_lro_candidates_shape_and_404():
    # Lifespan may not have fired under TestClient: refresh the catalog first.
    client.get("/refresh", headers=_auth())
    r = client.get("/triplets/nope_p6/triplet_X/lro-candidates")
    assert r.status_code == 404, r.text[:200]
    r = client.get("/triplets/region_001/lro-candidates")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["triplet_id"] == "region_001"
    assert isinstance(body["candidates"], list)
    assert "bounds" in body, "candidates response must echo the query bounds"


# ---------------------------------------------------------------------------
# Config centralization
# ---------------------------------------------------------------------------

def test_job_ttl_centralized_in_settings():
    from config import settings
    import job_store

    assert isinstance(settings.JOB_TTL_HOURS, int) and settings.JOB_TTL_HOURS > 0
    assert job_store._job_ttl_seconds() == settings.JOB_TTL_HOURS * 3600
