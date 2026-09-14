"""Delete endpoints: ingest history + dataset bundles."""
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

_TEST_JWT_SECRET = "test-jwt-secret-for-delete-tests-only"
os.environ.setdefault("JWT_SECRET_KEY", _TEST_JWT_SECRET)
try:
    from config import settings as _settings

    if not _settings.JWT_SECRET_KEY:
        _settings.JWT_SECRET_KEY = _TEST_JWT_SECRET
except Exception:
    pass

from routers import ingest as ingest_router  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from data import loader as _loader

    # Isolate on-disk roots per test.
    triplets = tmp_path / "processed_triplets"
    triplets.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "matches").mkdir()
    (data_dir / "user_triplets.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(_loader, "PROCESSED_TRIPLETS_DIR", str(triplets))
    monkeypatch.setattr(_loader, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(ingest_router, "_PROCESSED_TRIPLETS", triplets)
    monkeypatch.setattr(ingest_router, "_UPLOAD_ROOT", tmp_path / ".uploads")

    from main import app

    with TestClient(app) as c:
        yield c


def _auth(client):
    email = f"del-{uuid.uuid4().hex[:8]}@example.com"
    res = client.post(
        "/auth/register",
        json={"name": "T", "email": email, "password": "correct-horse-123"},
    )
    assert res.status_code == 201, f"auth bootstrap failed: {res.status_code} {res.text[:200]}"
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def test_delete_finished_job_removes_record_and_upload_dir(client):
    h = _auth(client)
    upload_dir = Path(ingest_router._UPLOAD_ROOT) / "job1"
    upload_dir.mkdir(parents=True)
    (upload_dir / "upload_0.zip").write_text("x")
    ingest_router._jobs["job1"] = {
        "job_id": "job1", "status": "completed", "stage": "done",
        "progress_pct": 100.0, "started_at": None, "completed_at": None,
        "log_lines": [], "error": None, "summary": "", "triplets": [],
        "upload_dir": str(upload_dir), "config": {},
    }
    try:
        r = client.delete("/api/ingest/jobs/job1", headers=h)
        assert r.status_code == 200, r.text[:300]
        assert "job1" not in ingest_router._jobs
        assert not upload_dir.exists()
    finally:
        ingest_router._jobs.pop("job1", None)


def test_delete_running_job_conflicts_and_bulk_mixed(client):
    h = _auth(client)
    ingest_router._jobs["run1"] = {
        "job_id": "run1", "status": "running", "stage": "x",
        "progress_pct": 10.0, "started_at": None, "completed_at": None,
        "log_lines": [], "error": None, "summary": "", "triplets": [],
        "upload_dir": None, "config": {},
    }
    ingest_router._jobs["done1"] = {
        "job_id": "done1", "status": "failed", "stage": "x",
        "progress_pct": 100.0, "started_at": None, "completed_at": None,
        "log_lines": [], "error": "boom", "summary": "", "triplets": [],
        "upload_dir": None, "config": {},
    }
    try:
        r = client.delete("/api/ingest/jobs/run1", headers=h)
        assert r.status_code == 409, r.text[:200]
        r = client.request(
            "DELETE", "/api/ingest/jobs", headers={**h, "Content-Type": "application/json"},
            json={"job_ids": ["done1", "missing", "run1"]},
        )
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["deleted"] == ["done1"]
        assert "missing" in body["errors"] and "run1" in body["errors"]
    finally:
        ingest_router._jobs.pop("run1", None)
        ingest_router._jobs.pop("done1", None)


def test_delete_dataset_generated_ok_curated_guarded(client, tmp_path, monkeypatch):
    from data import loader as _loader

    h = _auth(client)
    triplets = Path(_loader.PROCESSED_TRIPLETS_DIR)
    reg = triplets / "region_auto_900"
    reg.mkdir()
    (reg / "manifest.json").write_text(
        '{"region_id": "region_auto_900", "bounds": {"west_lon": 0, "east_lon": 1, "south_lat": 0, "north_lat": 1}}',
        encoding="utf-8",
    )
    (reg / "ohrc_512.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    r = client.delete("/triplets/region_auto_900", headers=h)
    assert r.status_code == 200, r.text[:300]
    assert not reg.exists()

    # Curated seeds need ?force=true.
    r = client.delete("/triplets/region_001", headers=h)
    assert r.status_code == 400, r.text[:200]
    assert "force=true" in r.text

    # Traversal rejected.
    r = client.delete("/triplets/..%2Fetc", headers=h)
    assert r.status_code in (400, 404)
