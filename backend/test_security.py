"""
test_security.py — Step 12 verification: TestClient auth + traversal + upload tests.

Run from backend/:  pytest test_security.py -v
Env is pinned BEFORE the app import so settings (JWT secret, rate limit)
are deterministic for the whole session.
"""

import io
import json
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-that-is-long-enough-for-the-32-char-minimum-0123456789",
)
os.environ.setdefault("ENVIRONMENT", "test")

from fastapi.testclient import TestClient  # noqa: E402

import auth_store  # noqa: E402
import rate_limit  # noqa: E402
from config import Settings, settings  # noqa: E402
from main import app  # noqa: E402
from data import loader  # noqa: E402
from routers.auth import limiter as auth_limiter  # noqa: E402


def _png_bytes(size: int = 32, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def isolated_users(tmp_path, monkeypatch):
    """Point the JSON user store at a temp file (never touches real users)."""
    monkeypatch.setattr(auth_store, "USERS_FILE", tmp_path / "users.json")
    auth_store._reset_for_tests()
    yield tmp_path
    auth_store._reset_for_tests()


@pytest.fixture()
def fresh_limits():
    """Clear SlowAPI counters + lockout memory so tests are independent."""
    try:
        auth_limiter.reset()
    except Exception:
        pass
    rate_limit._reset_for_tests()
    yield
    try:
        auth_limiter.reset()
    except Exception:
        pass
    rate_limit._reset_for_tests()


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def clean_runs():
    """Remove any dynamic_runs created during a test."""
    runs_dir = Path(loader.DATA_DIR) / "dynamic_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    before = set(os.listdir(runs_dir))
    yield runs_dir
    for entry in set(os.listdir(runs_dir)) - before:
        import shutil

        shutil.rmtree(runs_dir / entry, ignore_errors=True)


def _register(client, email: str, password: str = "correct-horse-123") -> dict:
    r = client.post(
        "/auth/register",
        json={"name": "Tester", "email": email, "password": password},
    )
    assert r.status_code == 201, r.text[:300]
    return r.json()


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# JWT secret has no default (fail closed)
# ---------------------------------------------------------------------------

def test_jwt_secret_has_no_default():
    src = (Path(__file__).resolve().parent / "config.py").read_text(encoding="utf-8")
    assert "change-in-production" not in src, "embedded dev secret must be gone"
    blank = Settings(_env_file=None, JWT_SECRET_KEY=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError):
        blank.require_jwt_secret()
    assert len(settings.require_jwt_secret()) >= 32


# ---------------------------------------------------------------------------
# Auth round-trip + protected endpoints
# ---------------------------------------------------------------------------

def test_auth_register_login_me_roundtrip(client, isolated_users, fresh_limits):
    email = f"roundtrip-{uuid.uuid4().hex[:8]}@example.com"
    reg = _register(client, email)
    token = reg["access_token"]
    assert reg["user"]["email"] == email
    assert "password_hash" not in json.dumps(reg)

    me = client.get("/auth/me", headers=_auth_header(token))
    assert me.status_code == 200
    assert me.json()["email"] == email

    login = client.post("/auth/login", json={"email": email, "password": "correct-horse-123"})
    assert login.status_code == 200
    assert login.json()["access_token"]

    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers=_auth_header("bogus")).status_code == 401


def test_bcrypt_cost_is_12(client, isolated_users, fresh_limits):
    email = f"rounds-{uuid.uuid4().hex[:8]}@example.com"
    _register(client, email)
    stored = json.loads((isolated_users / "users.json").read_text(encoding="utf-8"))
    pw_hash = next(u for u in stored if u["email"] == email)["password_hash"]
    assert pw_hash.startswith("$2b$12$"), f"expected bcrypt cost 12, got {pw_hash[:7]}"


def test_protected_endpoints_require_auth(client, clean_runs, fresh_limits):
    ref = _png_bytes()
    # POST /register without a token -> 401 (nothing written).
    r = client.post(
        "/register",
        files={
            "source_file": ("src.png", ref, "image/png"),
            "reference_file": ("ref.png", ref, "image/png"),
        },
        data={"source_sensor": "OHRC", "reference_sensor": "TMC", "method": "cfog"},
    )
    assert r.status_code == 401, r.text[:200]
    # GET /refresh without a token -> 401.
    assert client.get("/refresh").status_code == 401


def test_refresh_with_auth(client, isolated_users, fresh_limits):
    email = f"refresh-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    r = client.get("/refresh", headers=_auth_header(token))
    assert r.status_code == 200
    assert r.json()["status"] == "refreshed"


# ---------------------------------------------------------------------------
# Lockout + rate limits on /auth/*
# ---------------------------------------------------------------------------

def test_login_lockout_after_repeated_failures(client, isolated_users, fresh_limits):
    email = f"lockout-{uuid.uuid4().hex[:8]}@example.com"
    _register(client, email)
    for _ in range(5):
        r = client.post("/auth/login", json={"email": email, "password": "wrong-pass"})
        assert r.status_code == 401
    # 6th attempt — even the RIGHT password is refused while locked.
    r = client.post("/auth/login", json={"email": email, "password": "correct-horse-123"})
    assert r.status_code == 429, r.text[:200]
    assert "Retry-After" in r.headers


def test_auth_register_rate_limited(client, isolated_users, fresh_limits, monkeypatch):
    # Pin a tight limit for determinism regardless of session import order.
    from config import settings as _settings

    monkeypatch.setattr(_settings, "AUTH_RATE_LIMIT", "5/minute")
    statuses = []
    for i in range(7):
        r = client.post(
            "/auth/register",
            json={"name": "R", "email": f"rl-{uuid.uuid4().hex[:8]}@example.com",
                  "password": "correct-horse-123"},
        )
        statuses.append(r.status_code)
    assert 429 in statuses, f"expected a 429 among {statuses}"


# ---------------------------------------------------------------------------
# Traversal + FileResponse allowlist
# ---------------------------------------------------------------------------

def test_images_traversal_blocked(client):
    for target in (
        "/images/ohrc/..%2F..%2Fetc%2Fhostname",
        "/images/ohrc/..%2F..%2F..%2Fetc%2Fpasswd",
        "/images/dem/%2E%2E%2Fusers.json",
    ):
        r = client.get(target)
        assert r.status_code == 404, f"{target} -> {r.status_code}"
        assert "root:" not in r.text, "file content must never leak"


def test_file_response_allowlist_unit(tmp_path):
    from uploads import check_file_response_allowed

    roots = [tmp_path / "allowed"]
    (roots[0]).mkdir(parents=True)
    ok_file = roots[0] / "tile.png"
    ok_file.write_bytes(b"PNG")
    assert check_file_response_allowed(ok_file, roots, {".png"}).name == "tile.png"

    secret = tmp_path / "secret.txt"
    secret.write_text("root:x:0:0")
    with pytest.raises(Exception) as exc:
        check_file_response_allowed(secret, roots, {".png", ".txt"})
    assert getattr(exc.value, "status_code", 404) == 404
    # Symlink escape from inside the root is also refused.
    link = roots[0] / "evil.png"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(Exception) as exc2:
        check_file_response_allowed(link, roots, {".png"})
    assert getattr(exc2.value, "status_code", 404) == 404


# ---------------------------------------------------------------------------
# Upload validation: extensions, DEM gate, size caps, no residue
# ---------------------------------------------------------------------------

def test_register_rejects_evil_extension_without_writing(client, clean_runs, isolated_users,
                                                         fresh_limits):
    email = f"evil-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    before = set(os.listdir(clean_runs))
    r = client.post(
        "/register",
        headers=_auth_header(token),
        files={
            "source_file": ("evil.php", b"<?php echo 'pwn';", "application/x-php"),
            "reference_file": ("ref.png", _png_bytes(), "image/png"),
        },
        data={"source_sensor": "OHRC", "reference_sensor": "TMC", "method": "cfog"},
    )
    assert r.status_code == 415, r.text[:200]
    assert set(os.listdir(clean_runs)) == before


def test_register_validates_dem_upload(client, clean_runs, isolated_users, fresh_limits):
    email = f"dem-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    before = set(os.listdir(clean_runs))
    r = client.post(
        "/register",
        headers=_auth_header(token),
        files={
            "source_file": ("src.png", _png_bytes(), "image/png"),
            "reference_file": ("ref.png", _png_bytes(), "image/png"),
            "dem_file": ("evil.exe", b"MZ", "application/octet-stream"),
        },
        data={"source_sensor": "OHRC", "reference_sensor": "TMC", "method": "cfog"},
    )
    assert r.status_code == 415, r.text[:200]
    assert set(os.listdir(clean_runs)) == before


def test_oversize_upload_rejected_without_disk_write(client, clean_runs, isolated_users,
                                                     fresh_limits, monkeypatch):
    import main as app_main

    monkeypatch.setattr(app_main, "max_upload_bytes", lambda: 1024)  # 1 KB cap
    email = f"big-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    before = set(os.listdir(clean_runs))
    r = client.post(
        "/register",
        headers=_auth_header(token),
        files={
            "source_file": ("src.png", _png_bytes(), "image/png"),
            "reference_file": ("ref.png", _png_bytes(), "image/png"),
        },
        data={"source_sensor": "OHRC", "reference_sensor": "TMC", "method": "cfog"},
    )
    assert r.status_code == 413, r.text[:200]
    assert set(os.listdir(clean_runs)) == before, "oversize probe must leave no residue"


def test_100mb_upload_rejected(client, clean_runs, isolated_users, fresh_limits):
    email = f"huge-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    before = set(os.listdir(clean_runs))
    blob = b"\x00" * (100 * 1024 * 1024)
    r = client.post(
        "/register",
        headers=_auth_header(token),
        files={
            "source_file": ("big.png", blob, "image/png"),
            "reference_file": ("ref.png", _png_bytes(), "image/png"),
        },
        data={"source_sensor": "OHRC", "reference_sensor": "TMC", "method": "cfog"},
    )
    del blob
    assert r.status_code == 413, r.text[:200]
    assert set(os.listdir(clean_runs)) == before


# ---------------------------------------------------------------------------
# Authenticated /register success path (validation + matcher + threadpool)
# ---------------------------------------------------------------------------

def test_register_success_roundtrip_with_auth(client, clean_runs, isolated_users,
                                              fresh_limits):
    email = f"ok-{uuid.uuid4().hex[:8]}@example.com"
    token = _register(client, email)["access_token"]
    rng = np.random.RandomState(3)
    img = (rng.rand(128, 128) * 255).astype(np.uint8)
    for cx, cy, rad in ((64, 64, 22), (32, 96, 12), (100, 32, 10)):
        cv2.circle(img, (cx, cy), rad, 230, -1)
        cv2.circle(img, (cx, cy), max(2, rad - 4), 60, -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    payload = buf.tobytes()
    r = client.post(
        "/register",
        headers=_auth_header(token),
        files={
            "source_file": ("src.png", payload, "image/png"),
            "reference_file": ("ref.png", payload, "image/png"),
        },
        data={"source_sensor": "TMC", "reference_sensor": "TMC", "method": "cfog"},
        timeout=300,
    )
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    assert body["status"] == "success"
    assert body["homography"] is not None


# ---------------------------------------------------------------------------
# Users live in Postgres when configured (sqlite proves the DB path)
# ---------------------------------------------------------------------------

def test_users_db_backend_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("USERS_DATABASE_URL", f"sqlite:///{db_path}")
    auth_store._reset_for_tests()
    try:
        assert auth_store.using_database() is True
        created = auth_store.create_user({
            "id": "u-1", "name": "Db User", "email": "db@example.com",
            "password_hash": "x", "created_at": "now",
        })
        assert created["email"] == "db@example.com"
        assert auth_store.find_user_by_email("DB@EXAMPLE.COM")["id"] == "u-1"
        assert auth_store.find_user_by_id("u-1")["name"] == "Db User"
    finally:
        monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
        auth_store._reset_for_tests()


def test_run_ttl_purge(tmp_path):
    from uploads import purge_expired_runs

    runs = tmp_path / "dynamic_runs"
    old = runs / "old_run"
    new = runs / "new_run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "f.txt").write_text("x")
    (new / "f.txt").write_text("y")
    import time

    ancient = time.time() - 100 * 3600
    os.utime(old, (ancient, ancient))
    assert purge_expired_runs(tmp_path, ttl_hours=24) == 1
    assert not old.exists() and new.exists()
