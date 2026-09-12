"""
tests/test_lro_auto_discover_cli.py — CLI wiring test for prepare_lro_nac_pair.py --auto-discover (Step 4)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline" / "scripts"))

from prepare_lro_nac_pair import main


def test_cli_auto_discover_flag_invokes_fetch_and_prepare(tmp_path):
    """Confirm --auto-discover flag invokes fetch_and_prepare_lro_nac with correct arguments."""
    with patch("lro_ode_client.fetch_and_prepare_lro_nac") as mock_fetch:
        main([
            "--regions", "region_001",
            "--auto-discover",
            "--output_dir", str(tmp_path),
        ])

        assert mock_fetch.called
        call_kwargs = mock_fetch.call_args.kwargs
        assert call_kwargs.get("region_id") == "region_001"
        assert call_kwargs.get("output_dir") == tmp_path / "region_001"
        assert "region_bounds" in call_kwargs
        bounds = call_kwargs["region_bounds"]
        assert all(k in bounds for k in ("west_lon", "east_lon", "south_lat", "north_lat"))


def test_cli_manual_default_does_not_invoke_fetch_and_prepare():
    """Confirm omission of --auto-discover retains original prepare_pair_for_region manual path."""
    with patch("lro_ode_client.fetch_and_prepare_lro_nac") as mock_fetch, \
         patch("prepare_lro_nac_pair.prepare_pair_for_region") as mock_prepare:
        main([
            "--regions", "region_001",
            "--raw_nac_img", "dummy_nac.png",
            "--raw_nac_lbl", "dummy_nac.lbl",
        ])

        assert not mock_fetch.called
        assert mock_prepare.called
        call_kwargs = mock_prepare.call_args.kwargs
        assert call_kwargs.get("region_id") == "region_001"
        assert call_kwargs.get("raw_nac_image") == "dummy_nac.png"
        assert call_kwargs.get("raw_nac_label") == "dummy_nac.lbl"
