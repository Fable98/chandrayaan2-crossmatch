"""
tests/test_borrowed_features.py — Verification for the three borrowed items.

1. Job log streaming (backend registration router): ring buffer semantics +
   GET /logs/{job_id} contract. Mirrors the ingest router's log_lines pattern.
2. Displacement-vector quiver QA: file output on valid input, None on
   degenerate input, arrows actually drawn (non-blank PNG).
3. Synthetic-GT CI regression harness: known affine warp of a real tile must
   be recovered coarsely (status success, >=4 inliers, corner error <=12px).

   *** The synthetic check is a SOFTWARE SMOKE TEST, not benchmark evidence:
   recovering a known self-warp says nothing about cross-sensor robustness.
   See docs/benchmark_sun_gap.md for real orbital evidence. ***
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# 1. Registration job logs
# ---------------------------------------------------------------------------

def test_job_log_ring_caps_and_slices():
    from routers.registration import job_manager
    jid = "test-log-ring"
    job_manager.create_job(jid, "test")
    try:
        for i in range(250):
            job_manager.append_log(jid, f"line {i}")
        data = job_manager.get_logs(jid)
        assert data["total"] == 200
        assert data["lines"][0].endswith("line 50")
        tail = job_manager.get_logs(jid, after=195)
        assert tail["after"] == 195 and len(tail["lines"]) == 5
        assert job_manager.get_logs("no-such-job") is None
    finally:
        job_manager.jobs.pop(jid, None)


def test_job_logs_endpoint_contract():
    from fastapi.testclient import TestClient
    from routers.registration import job_manager, router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    jid = "test-log-endpoint"
    job_manager.create_job(jid, "test")
    try:
        job_manager.append_log(jid, "hello worker")
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/api/registration/logs/does-not-exist").status_code == 404
            r = client.get(f"/api/registration/logs/{jid}")
            assert r.status_code == 200
            body = r.json()
            assert body["job_id"] == jid and body["total"] == 1
            assert body["lines"][0].endswith("hello worker")
            r2 = client.get(f"/api/registration/logs/{jid}?after=1")
            assert r2.json()["lines"] == []
    finally:
        job_manager.jobs.pop(jid, None)


# ---------------------------------------------------------------------------
# 2. Quiver QA
# ---------------------------------------------------------------------------

def test_quiver_writes_png_and_rejects_degenerate(tmp_path):
    from quiver import create_displacement_quiver
    rng = np.random.RandomState(7)
    src = rng.rand(12, 2) * 400 + 50
    H = np.array([[1, 0.02, 6.0], [-0.01, 1, -4.0], [0, 0, 1]])
    ones = np.ones((len(src), 1))
    dst = (H @ np.hstack([src, ones]).T).T
    dst = dst[:, :2] / dst[:, 2:3] + rng.randn(*src.shape) * 0.4

    out = tmp_path / "q.png"
    assert create_displacement_quiver(src, dst, H, (512, 512), out) == str(out)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 100
    # Non-blank: quiver draws colored arrows on white axes.
    assert float(np.std(img)) > 1.0

    assert create_displacement_quiver(np.zeros((0, 2)), np.zeros((0, 2)), H,
                                      (512, 512), tmp_path / "e.png") is None


def test_register_module_quiver_helper():
    from register import create_displacement_quiver as reg_q
    assert callable(reg_q)


# ---------------------------------------------------------------------------
# 3. Synthetic-GT CI regression (SMOKE ONLY — see module docstring)
# ---------------------------------------------------------------------------

def test_synthetic_known_warp_recovered(tmp_path):
    from synthetic_gt_check import build_synthetic_pair, corner_error
    from matcher_cfog import match_images_cfog

    tile = (REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
            / "region_001" / "ohrc_512.png")
    src = cv2.imread(str(tile), cv2.IMREAD_GRAYSCALE)
    assert src is not None
    tgt, H_gt = build_synthetic_pair(src)
    p_src, p_tgt = tmp_path / "s.png", tmp_path / "t.png"
    cv2.imwrite(str(p_src), src)
    cv2.imwrite(str(p_tgt), tgt)

    res = match_images_cfog(str(p_src), str(p_tgt), output_dir=str(tmp_path / "w"),
                            source_sensor="OHRC", reference_sensor="OHRC",
                            explicit_gsd1=5.0, explicit_gsd2=5.0)
    assert res.get("status") == "success"
    met = res.get("metrics") or {}
    assert met.get("inlier_count", 0) >= 4
    err = corner_error(np.asarray(res["homography"]), H_gt, src.shape)
    assert err <= 12.0, f"synthetic warp not recovered: corner err {err:.2f}px"
