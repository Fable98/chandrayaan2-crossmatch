"""
tests/test_frontend_contract.py — Step 13 backend-side contract tests.

  * moon-points without product bounds => latitude/longitude null +
    georeferenced=false (the 336+fx demo patch is gone); with bounds =>
    real geo.py coordinates and georeferenced=true.
  * generated TS contract (backend-types.ts) matches the live openapi.json
    (runs gen-openapi.mjs --check; skipped when node is unavailable).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-that-is-long-enough-for-the-32-char-minimum-0123456789",
)
os.environ.setdefault("ENVIRONMENT", "test")


def _mem_job_manager():
    from routers import registration as reg_router
    from job_store import MemoryJobStore

    mgr = reg_router.JobManager(store=MemoryJobStore())
    return reg_router, mgr


def test_moon_points_without_bounds_are_not_georeferenced(monkeypatch):
    reg_router, mgr = _mem_job_manager()
    monkeypatch.setattr(reg_router, "job_manager", mgr)
    mgr.create_job("job-nobounds", "registration")
    mgr.update_job(
        "job-nobounds",
        status="success",
        result={
            "final_rmse_pixels": 0.5,
            "filtered_ref_pts": [[10.0, 20.0], [30.0, 40.0]],
            "filtered_src_pts": [[11.0, 21.0], [31.0, 41.0]],
            # No "bounds" key anywhere: coordinates must NOT be invented.
        },
    )
    import asyncio

    resp = asyncio.run(reg_router.get_moon_points("job-nobounds"))
    assert resp.georeferenced is False
    assert resp.georef_note is not None
    assert len(resp.points) == 2
    for pt in resp.points:
        assert pt.latitude is None and pt.longitude is None
        assert pt.georeferenced is False
        assert pt.pixel_x is not None and pt.pixel_y is not None


def test_moon_points_with_bounds_use_geo_py(monkeypatch):
    reg_router, mgr = _mem_job_manager()
    monkeypatch.setattr(reg_router, "job_manager", mgr)
    mgr.create_job("job-bounds", "registration")
    mgr.update_job(
        "job-bounds",
        status="success",
        result={
            "final_rmse_pixels": 0.5,
            "filtered_ref_pts": [[0.0, 0.0], [512.0, 512.0]],
            "filtered_src_pts": [[0.0, 0.0], [512.0, 512.0]],
            "bounds": {
                "west_lon": 336.0,
                "east_lon": 337.0,
                "south_lat": -4.0,
                "north_lat": -3.0,
            },
            "width": 512.0,
            "height": 512.0,
        },
    )
    import asyncio

    resp = asyncio.run(reg_router.get_moon_points("job-bounds"))
    assert resp.georeferenced is True
    assert resp.georef_note is None
    # geo.py shared affine: TL -> (north, west), BR -> (south, east).
    assert resp.points[0].latitude == pytest.approx(-3.0)
    assert resp.points[0].longitude == pytest.approx(336.0)
    assert resp.points[1].latitude == pytest.approx(-4.0)
    assert resp.points[1].longitude == pytest.approx(337.0)
    assert all(pt.georeferenced for pt in resp.points)


def test_generated_ts_contract_matches_openapi():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    env = dict(os.environ)
    spec_path = REPO_ROOT / "backend" / "test_openapi_tmp.json"
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from main import app

    spec_path.write_text(json.dumps(app.openapi()), encoding="utf-8")
    try:
        env["OPENAPI_FILE"] = str(spec_path)
        proc = subprocess.run(
            [node, "scripts/gen-openapi.mjs", "--check"],
            cwd=str(REPO_ROOT / "lunar-frontend"),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"backend-types.ts drifted from openapi.json:\n{proc.stdout}\n{proc.stderr}\n"
            "Run `npm run gen:api` in lunar-frontend and commit the result."
        )
    finally:
        spec_path.unlink(missing_ok=True)
