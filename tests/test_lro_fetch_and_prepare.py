"""
tests/test_lro_fetch_and_prepare.py — Unit tests for LRO NAC fetch_and_prepare_lro_nac (Step 3b/3c)
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from lro_ode_client import (
    MAX_DOWNLOAD_BYTES,
    MIN_OVERLAP_THRESHOLD,
    _stream_download,
    fetch_and_prepare_lro_nac,
)

SAMPLE_LBL_CONTENT = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 512
FILE_RECORDS = 513
LABEL_RECORDS = 1
^IMAGE = 2

DATA_SET_ID = "LRO-L-LROC-3-CDR-V1.0"
PRODUCT_ID = "M1417670274LC"
INSTRUMENT_ID = LROC
FRAME_ID = LEFT
START_TIME = 2022-09-13T01:03:26.869
INCIDENCE_ANGLE = 5.82 <deg>
EMISSION_ANGLE = 1.71 <deg>
SOLAR_AZIMUTH_ANGLE = 85.4 <deg>

OBJECT = IMAGE_MAP_PROJECTION
  MAP_SCALE = 0.914 <m>
  MINIMUM_LATITUDE = -3.81
  MAXIMUM_LATITUDE = -2.18
  WESTERNMOST_LONGITUDE = 336.37
  EASTERNMOST_LONGITUDE = 336.64
END_OBJECT = IMAGE_MAP_PROJECTION

OBJECT = IMAGE
  LINES = 16
  LINE_SAMPLES = 16
  SAMPLE_BITS = 8
  SAMPLE_TYPE = UNSIGNED_INTEGER
END_OBJECT = IMAGE
END
"""

TEST_REGION_BOUNDS = {
    "west_lon": 336.48,
    "east_lon": 336.58,
    "south_lat": -3.51,
    "north_lat": -3.42,
}


def test_prerequisite_check_missing_region_returns_none(tmp_path):
    """If region_id is not ingested into processed_triplets, return None immediately with error."""
    out_dir = tmp_path / "lro_nac_real" / "nonexistent_region"
    res = fetch_and_prepare_lro_nac(
        region_bounds=TEST_REGION_BOUNDS,
        region_id="nonexistent_region",
        output_dir=out_dir,
    )
    assert res is None
    # Ensure no network or output dir activity
    assert not out_dir.exists()


def test_size_guard_aborts_and_deletes_partial_download(tmp_path):
    """Test that stream download aborts and cleans up when size exceeds MAX_DOWNLOAD_BYTES."""
    dest = tmp_path / "too_large.bin"

    # Generate a mock response that streams more than max_bytes
    chunk_1mb = b"X" * (1024 * 1024)
    chunks = [chunk_1mb] * 10  # 10 MB total

    class MockStream:
        def __init__(self, chunk_list):
            self._chunks = list(chunk_list)

        def read(self, size):
            if self._chunks:
                return self._chunks.pop(0)
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=MockStream(chunks)):
        # Set max_bytes to 5 MB
        with pytest.raises(ValueError, match="Download exceeded unconditional size limit"):
            _stream_download(
                url="https://example.com/huge_file.img",
                dest_path=dest,
                max_bytes=5 * 1024 * 1024,
            )

    # Assert partial file was removed
    assert not dest.exists(), "Partial download was not deleted after exceeding size guard!"


def test_anti_fabrication_check_rejects_synthetic_proxy_and_preserves_preexisting(tmp_path):
    """
    If prepare_pair_for_region falls back to synthetic_ohrc_derived_proxy,
    fetch_and_prepare_lro_nac must raise an explicit error AND preserve pre-existing files.
    """
    region_id = "test_synth_region"
    # Create valid processed_triplets prerequisite
    triplet_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / region_id
    triplet_dir.mkdir(parents=True, exist_ok=True)
    (triplet_dir / "manifest.json").write_text(json.dumps({"ohrc_product_id": "TEST_OHRC", "bounds": TEST_REGION_BOUNDS}))
    (triplet_dir / "ohrc_512.png").write_bytes(b"FAKE_OHRC_IMAGE")

    out_dir = tmp_path / "lro_nac_real" / region_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-existing file in out_dir
    pre_existing_file = out_dir / "pre_existing_benchmark.json"
    pre_existing_file.write_text(json.dumps({"original_data": "must_not_be_deleted"}))

    candidate = {
        "product_id": "M1417670274LC",
        "label_url": "https://example.com/test.lbl",
        "download_urls": ["https://example.com/test.img"],
        "footprint_bounds": {
            "west_lon": 336.30,
            "east_lon": 336.70,
            "south_lat": -3.60,
            "north_lat": -3.20,
        },
        "incidence_angle_deg": 5.82,
    }

    # Synthetic image raster (16x16 uint8)
    fake_pixels = np.zeros((16, 16), dtype=np.uint8).tobytes()

    def fake_stream(url, dest_path, max_bytes=None):
        if str(dest_path).endswith(".LBL"):
            dest_path.write_text(SAMPLE_LBL_CONTENT)
        else:
            dest_path.write_bytes(fake_pixels)
        return 100

    def fake_prepare_pair_synthetic(**kwargs):
        od = Path(kwargs["output_dir"])
        new_file = od / "newly_created_artifact.png"
        new_file.write_bytes(b"RUN_OUTPUT")
        return {
            "region_id": region_id,
            "reference_provenance": "synthetic_ohrc_derived_proxy",  # SYNTHETIC FALLBACK!
        }

    try:
        with patch("lro_ode_client.search_lro_nac_overlap", return_value=[candidate]), \
             patch("lro_ode_client._stream_download", side_effect=fake_stream), \
             patch("lro_ode_client.prepare_pair_for_region", side_effect=fake_prepare_pair_synthetic):

            with pytest.raises(RuntimeError, match="Anti-fabrication check failed"):
                fetch_and_prepare_lro_nac(
                    region_bounds=TEST_REGION_BOUNDS,
                    region_id=region_id,
                    output_dir=out_dir,
                )

        # Pre-existing file MUST NOT be deleted
        assert pre_existing_file.exists(), "Pre-existing file was wrongly destroyed during failure cleanup!"
        assert json.loads(pre_existing_file.read_text())["original_data"] == "must_not_be_deleted"

        # Newly created artifact MUST have been cleaned up
        assert not (out_dir / "newly_created_artifact.png").exists(), "Newly created artifact was not cleaned up!"

    finally:
        # Cleanup test triplet directory
        import shutil
        shutil.rmtree(triplet_dir, ignore_errors=True)


def test_minimum_overlap_threshold_guard(tmp_path):
    """Candidate with overlap below MIN_OVERLAP_THRESHOLD returns None."""
    region_id = "test_low_overlap"
    triplet_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / region_id
    triplet_dir.mkdir(parents=True, exist_ok=True)
    (triplet_dir / "manifest.json").write_text(json.dumps({"bounds": TEST_REGION_BOUNDS}))
    (triplet_dir / "ohrc_512.png").write_bytes(b"FAKE_OHRC")

    out_dir = tmp_path / "lro_nac_real" / region_id

    # Candidate with barely 5% overlap
    candidate = {
        "product_id": "M_SLIGHT_OVERLAP",
        "label_url": "https://example.com/test.lbl",
        "download_urls": ["https://example.com/test.img"],
        "footprint_bounds": {
            "west_lon": 336.48,
            "east_lon": 336.485,  # 5% of target width
            "south_lat": -3.51,
            "north_lat": -3.42,
        },
    }

    try:
        with patch("lro_ode_client.search_lro_nac_overlap", return_value=[candidate]):
            res = fetch_and_prepare_lro_nac(
                region_bounds=TEST_REGION_BOUNDS,
                region_id=region_id,
                output_dir=out_dir,
            )
            assert res is None
    finally:
        import shutil
        shutil.rmtree(triplet_dir, ignore_errors=True)


def test_fetch_and_prepare_lro_nac_success(tmp_path):
    """Successful end-to-end run returns manifest with real_downloaded_cdr provenance."""
    region_id = "test_success_region"
    triplet_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / region_id
    triplet_dir.mkdir(parents=True, exist_ok=True)
    (triplet_dir / "manifest.json").write_text(json.dumps({"bounds": TEST_REGION_BOUNDS}))
    (triplet_dir / "ohrc_512.png").write_bytes(b"FAKE_OHRC")

    out_dir = tmp_path / "lro_nac_real" / region_id

    candidate = {
        "product_id": "M1417670274LC",
        "label_url": "https://example.com/M1417670274LC.LBL",
        "download_urls": ["https://example.com/M1417670274LC.IMG"],
        "footprint_bounds": {
            "west_lon": 336.30,
            "east_lon": 336.70,
            "south_lat": -3.60,
            "north_lat": -3.20,
        },
        "incidence_angle_deg": 5.82,
    }

    fake_pixels = np.zeros((16, 16), dtype=np.uint8).tobytes()

    def fake_stream(url, dest_path, max_bytes=None):
        if str(dest_path).endswith(".LBL"):
            dest_path.write_text(SAMPLE_LBL_CONTENT)
        else:
            dest_path.write_bytes(fake_pixels)
        return 256

    def fake_prepare_pair_success(**kwargs):
        return {
            "region_id": region_id,
            "reference_provenance": "real_downloaded_cdr",
            "reference_sensor": "LRO_NAC",
        }

    try:
        with patch("lro_ode_client.search_lro_nac_overlap", return_value=[candidate]), \
             patch("lro_ode_client._stream_download", side_effect=fake_stream), \
             patch("lro_ode_client.prepare_pair_for_region", side_effect=fake_prepare_pair_success):

            manifest = fetch_and_prepare_lro_nac(
                region_bounds=TEST_REGION_BOUNDS,
                region_id=region_id,
                output_dir=out_dir,
                incidence_angle=5.82,
            )

            assert manifest is not None
            assert manifest["reference_provenance"] == "real_downloaded_cdr"

    finally:
        import shutil
        shutil.rmtree(triplet_dir, ignore_errors=True)
