"""test_ingest_scoping.py — Tests run-level staging isolation and staleness prevention.

Verifies:
1. Each ingestion run is strictly isolated to its own staging directory.
2. Ingesting batch 2 does not leak or reuse products from batch 1.
3. Ingesting an upload with 0 overlapping triplets honestly reports 0 triplets
   and never silently falls back to stale triplets from past runs.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline" / "scripts"))

import ingest_and_prepare as iap


def _create_pds4_xml(product_id: str, sensor: str, west: float, east: float, south: float, north: float) -> str:
    sensor_map = {
        "OHRC": ("OHRC", "ch2_ohr_"),
        "TMC-2": ("TMC-2", "ch2_tmc_"),
        "IIRS": ("IIRS", "ch2_iir_"),
    }
    inst_name, _ = sensor_map[sensor]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <logical_identifier>urn:isro:ch2:{product_id}</logical_identifier>
        <title>{sensor} Observation {product_id}</title>
    </Identification_Area>
    <Observation_Area>
        <Observing_System>
            <Observing_System_Component>
                <name>{inst_name}</name>
                <type>Instrument</type>
            </Observing_System_Component>
        </Observing_System>
    </Observation_Area>
    <cart:Spatial_Domain xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
        <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>{west}</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>{east}</cart:east_bounding_coordinate>
            <cart:south_bounding_coordinate>{south}</cart:south_bounding_coordinate>
            <cart:north_bounding_coordinate>{north}</cart:north_bounding_coordinate>
        </cart:Bounding_Coordinates>
    </cart:Spatial_Domain>
</Product_Observational>
"""


def _create_zip_with_labels(zip_path: Path, labels: dict[str, str]):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname, content in labels.items():
            zf.writestr(fname, content)


def test_ingest_staging_scoping_prevents_staleness(tmp_path, monkeypatch):
    output_dir = tmp_path / "processed_triplets"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "user_triplets.json"

    # Mock stage_process_triplets so we test Stages 1, 2, 3 and 6 end-to-end
    # without needing full multi-gigabyte GDAL rasters.
    def mock_stage_process_triplets(triplets, output_dir, **kwargs):
        results = []
        for i, t in enumerate(triplets):
            rid = f"region_auto_{i+1:03d}"
            results.append({
                "region_id": rid,
                "triplet": t,
                "success": True,
                "manifest": {
                    "region_id": rid,
                    "bounds": t.get("bounds") or {
                        "west_lon": 0.0, "east_lon": 1.0, "south_lat": 0.0, "north_lat": 1.0
                    },
                }
            })
        return results

    monkeypatch.setattr(iap, "stage_process_triplets", mock_stage_process_triplets)

    # -------------------------------------------------------------
    # Batch 1: Region at lon 20, lat 10
    # -------------------------------------------------------------
    batch1_dir = tmp_path / "upload_batch1"
    b1_labels = {
        "ohrc.xml": _create_pds4_xml("ch2_ohr_batch1_001", "OHRC", 20.0, 20.5, 10.0, 10.5),
        "tmc.xml": _create_pds4_xml("ch2_tmc_batch1_001", "TMC-2", 20.0, 20.5, 10.0, 10.5),
        "iirs.xml": _create_pds4_xml("ch2_iir_batch1_001", "IIRS", 20.0, 20.5, 10.0, 10.5),
    }
    _create_zip_with_labels(batch1_dir / "upload_0.zip", b1_labels)

    # Ingest Batch 1
    ret1 = iap.main([
        str(batch1_dir),
        "--output-dir", str(output_dir),
        "--manifest", str(manifest_path),
        "--run-id", "job_batch1",
        "--containment", "0.8",
        "--no-matching",
        "--no-registration",
    ])
    assert ret1 == 0

    per_run_1 = json.loads((output_dir / ".last_run_triplets.json").read_text())
    assert per_run_1.get("run_id") == "job_batch1"
    triplets_1 = per_run_1.get("triplets", [])
    assert len(triplets_1) == 1
    assert triplets_1[0]["ohrc_product_id"] == "ch2_ohr_batch1_001"
    assert triplets_1[0]["tmc2_product_id"] == "ch2_tmc_batch1_001"
    assert triplets_1[0]["iirs_product_id"] == "ch2_iir_batch1_001"

    # -------------------------------------------------------------
    # Batch 2: Completely different region at lon 120, lat -40
    # -------------------------------------------------------------
    batch2_dir = tmp_path / "upload_batch2"
    b2_labels = {
        # Notice we purposefully name the zip upload_0.zip just like the web UI does!
        "ohrc.xml": _create_pds4_xml("ch2_ohr_batch2_999", "OHRC", 120.0, 120.5, -40.5, -40.0),
        "tmc.xml": _create_pds4_xml("ch2_tmc_batch2_999", "TMC-2", 120.0, 120.5, -40.5, -40.0),
        "iirs.xml": _create_pds4_xml("ch2_iir_batch2_999", "IIRS", 120.0, 120.5, -40.5, -40.0),
    }
    _create_zip_with_labels(batch2_dir / "upload_0.zip", b2_labels)

    # Ingest Batch 2
    ret2 = iap.main([
        str(batch2_dir),
        "--output-dir", str(output_dir),
        "--manifest", str(manifest_path),
        "--run-id", "job_batch2",
        "--containment", "0.8",
        "--no-matching",
        "--no-registration",
    ])
    assert ret2 == 0

    per_run_2 = json.loads((output_dir / ".last_run_triplets.json").read_text())
    assert per_run_2.get("run_id") == "job_batch2"
    triplets_2 = per_run_2.get("triplets", [])
    assert len(triplets_2) == 1

    # STRICT ASSERTIONS:
    # 1. Batch 2 result reflects ONLY batch 2's product IDs
    assert triplets_2[0]["ohrc_product_id"] == "ch2_ohr_batch2_999"
    assert triplets_2[0]["tmc2_product_id"] == "ch2_tmc_batch2_999"
    assert triplets_2[0]["iirs_product_id"] == "ch2_iir_batch2_999"

    # 2. Batch 1 product IDs did NOT leak into Batch 2
    for t in triplets_2:
        assert t["ohrc_product_id"] != "ch2_ohr_batch1_001"
        assert t["tmc2_product_id"] != "ch2_tmc_batch1_001"
        assert t["iirs_product_id"] != "ch2_iir_batch1_001"

    # -------------------------------------------------------------
    # Batch 3: Non-overlapping / incomplete upload (0 valid triplets)
    # -------------------------------------------------------------
    batch3_dir = tmp_path / "upload_batch3"
    b3_labels = {
        # Only an isolated OHRC, no TMC or IIRS
        "ohrc_lonely.xml": _create_pds4_xml("ch2_ohr_lonely_555", "OHRC", 50.0, 50.5, 0.0, 0.5),
    }
    _create_zip_with_labels(batch3_dir / "upload_0.zip", b3_labels)

    ret3 = iap.main([
        str(batch3_dir),
        "--output-dir", str(output_dir),
        "--manifest", str(manifest_path),
        "--run-id", "job_batch3",
        "--containment", "0.8",
        "--no-matching",
        "--no-registration",
    ])
    assert ret3 == 0

    per_run_3 = json.loads((output_dir / ".last_run_triplets.json").read_text())
    assert per_run_3.get("run_id") == "job_batch3"
    # MUST honestly report 0 triplets
    assert per_run_3.get("triplets") == []
    assert "0 new triplets found in this upload" in per_run_3.get("message", "")


@pytest.mark.asyncio
async def test_backend_ingest_no_stale_fallback_when_zero_triplets(tmp_path, monkeypatch):
    """The backend router must never fall back to user_triplets.json when a job yields 0 triplets."""
    import routers.ingest as ringest

    job_id = "testjob01"
    upload_dir = tmp_path / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Put a mock user_triplets.json with old historical triplets
    mock_processed = tmp_path / "processed_triplets"
    mock_processed.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ringest, "_PROCESSED_TRIPLETS", mock_processed)

    # Write a per_run file indicating 0 triplets were found in this upload
    per_run_file = mock_processed / ".last_run_triplets.json"
    per_run_file.write_text(json.dumps({
        "job_input_dir": str(upload_dir),
        "run_id": job_id,
        "triplets": [],
        "message": "0 new triplets found in this upload",
    }))

    # Write old triplets in user_triplets.json
    mock_pipeline_root = tmp_path / "pipeline"
    mock_pipeline_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ringest, "_PIPELINE_ROOT", mock_pipeline_root)
    (mock_pipeline_root / "user_triplets.json").write_text(json.dumps([
        {
            "region_id": "region_old_historical",
            "ohrc_product_id": "ch2_ohr_ancient",
            "tmc2_product_id": "ch2_tmc_ancient",
            "iirs_product_id": "ch2_iir_ancient",
            "overlap_triplet_pct": 90.0,
        }
    ]))

    ringest._jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "Starting",
        "progress_pct": 0.0,
        "started_at": None,
        "completed_at": None,
        "log_lines": [],
        "error": None,
        "summary": "",
        "triplets": [],
    }

    # Mock the subprocess execution to simply exit 0
    class MockProcess:
        stdout = None
        async def wait(self):
            return 0

    async def mock_subprocess_exec(*args, **kwargs):
        mp = MockProcess()
        async def _fake_iter():
            yield b"INGEST & PREPARE -- SUMMARY\n0 new triplets found in this upload\n"
        mp.stdout = _fake_iter()
        return mp

    monkeypatch.setattr("asyncio.create_subprocess_exec", mock_subprocess_exec)

    config = ringest.IngestConfig()
    await ringest._run_ingest_job(job_id, upload_dir, config)

    job = ringest._jobs[job_id]
    assert job["status"] == "completed"
    # MUST be strictly empty — NO silent fallback to region_old_historical!
    assert job["triplets"] == []

