"""
tests/test_lro_ode_client.py — Unit tests for LRO ODE REST Client (Step 1)
"""

import io
import json
import socket
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from lro_ode_client import (
    ODE_TIMEOUT_S,
    _extract_product_list,
    _parse_candidate_product,
    search_lro_nac_overlap,
)

SAMPLE_ODE_SUCCESS_JSON = {
    "ODEResults": {
        "Status": "Success",
        "Products": {
            "Product": [
                {
                    "pdsid": "M1417670274LC",
                    "LabelURL": "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0041/DATA/EDRNAC/2022256/M1417670274LC.LBL",
                    "Product_files": {
                        "Product_file": [
                            {
                                "URL": "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0041/DATA/EDRNAC/2022256/M1417670274LC.LBL",
                                "Type": "Label",
                            },
                            {
                                "URL": "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0041/DATA/EDRNAC/2022256/M1417670274LC.IMG",
                                "Type": "Data",
                            },
                        ]
                    },
                    "Westernmost_longitude": 336.37,
                    "Easternmost_longitude": 336.64,
                    "Minimum_latitude": -3.81,
                    "Maximum_latitude": -2.18,
                    "Incidence_angle": 5.82,
                    "Emission_angle": 1.71,
                },
                {
                    "pdsid": "M1413636095LC",
                    "LabelURL": "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0040/DATA/EDRNAC/2022208/M1413636095LC.LBL",
                    "Product_files": [
                        {
                            "URL": "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0040/DATA/EDRNAC/2022208/M1413636095LC.IMG",
                            "Type": "Data",
                        }
                    ],
                    "Westernmost_longitude": 336.35,
                    "Easternmost_longitude": 336.60,
                    "Minimum_latitude": -3.85,
                    "Maximum_latitude": -2.20,
                    "Incidence_angle": 51.74,
                },
            ]
        },
    }
}

SAMPLE_ODE_EMPTY_JSON = {
    "ODEResults": {
        "Status": "Success",
        "Products": {"Product": []},
    }
}

SAMPLE_ODE_MISSING_GEOMETRY_JSON = {
    "ODEResults": {
        "Status": "Success",
        "Products": {
            "Product": [
                {
                    "pdsid": "M_UNKNOWN_ORBIT",
                    "Product_files": [
                        {"URL": "https://example.com/data/M_UNKNOWN_ORBIT.LBL"},
                        {"URL": "https://example.com/data/M_UNKNOWN_ORBIT.IMG"},
                    ],
                }
            ]
        },
    }
}

TEST_BOUNDS = {
    "west_lon": 336.484646,
    "east_lon": 336.589455,
    "south_lat": -3.518776,
    "north_lat": -3.424168,
}


def _mock_response(json_data: dict, status: int = 200):
    raw_bytes = json.dumps(json_data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_resp.status = status
    return mock_resp


def test_search_lro_nac_overlap_success(tmp_path):
    """Test successful ODE query returns candidate list conforming to Step 1 contract."""
    mock_resp = _mock_response(SAMPLE_ODE_SUCCESS_JSON)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            product_type="EDRNAC",
            refresh_cache=True,
            cache_dir=tmp_path,
        )

        assert mock_urlopen.called
        req, kwargs = mock_urlopen.call_args
        assert kwargs.get("timeout") == ODE_TIMEOUT_S

        assert len(candidates) == 2

        # Step 1 contract: Every candidate MUST have product_id, label_url, download_urls, footprint_bounds
        c0 = candidates[0]
        assert c0["product_id"] == "M1417670274LC"
        assert c0["label_url"] == "https://pdsimage2.wr.usgs.gov/archive/lro-l-lroc-2-edr-v1.0/LROLRC_0041/DATA/EDRNAC/2022256/M1417670274LC.LBL"
        assert len(c0["download_urls"]) == 2
        assert c0["footprint_bounds"] == {
            "west_lon": 336.37,
            "east_lon": 336.64,
            "south_lat": -3.81,
            "north_lat": -2.18,
        }
        assert c0["incidence_angle_deg"] == pytest.approx(5.82)
        assert c0["emission_angle_deg"] == pytest.approx(1.71)


def test_search_lro_nac_overlap_empty_result(tmp_path):
    """ODE empty response returns empty list without raising."""
    mock_resp = _mock_response(SAMPLE_ODE_EMPTY_JSON)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
        )
        assert candidates == []


def test_search_lro_nac_overlap_missing_geometry(tmp_path):
    """Missing geometry/incidence must set footprint_bounds=None and not raise."""
    mock_resp = _mock_response(SAMPLE_ODE_MISSING_GEOMETRY_JSON)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
        )
        assert len(candidates) == 1
        c = candidates[0]
        assert c["product_id"] == "M_UNKNOWN_ORBIT"
        assert c["footprint_bounds"] is None
        assert "incidence_angle_deg" not in c
        assert c["label_url"] == "https://example.com/data/M_UNKNOWN_ORBIT.LBL"
        assert len(c["download_urls"]) == 2


def test_search_lro_nac_overlap_http_error(tmp_path):
    """HTTP errors return empty list and log warning rather than raising."""
    err = urllib.error.HTTPError("https://oderest.rsl.wustl.edu/", 500, "Internal Server Error", {}, io.BytesIO())
    with patch("urllib.request.urlopen", side_effect=err):
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
        )
        assert candidates == []


def test_search_lro_nac_overlap_timeout(tmp_path):
    """Timeouts return empty list and log warning rather than raising."""
    with patch("urllib.request.urlopen", side_effect=socket.timeout("Connection timed out")):
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
        )
        assert candidates == []


def test_search_lro_nac_overlap_malformed_json(tmp_path):
    """Malformed non-JSON body returns empty list."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html><head><title>502 Bad Gateway</title></head></html>"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        candidates = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
        )
        assert candidates == []


def test_search_lro_nac_overlap_cache_hit_and_refresh(tmp_path):
    """Cache loads existing queries without hitting network unless refresh_cache=True or TTL expired."""
    mock_resp = _mock_response(SAMPLE_ODE_SUCCESS_JSON)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        # First query: writes cache
        res1 = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=False,
            cache_dir=tmp_path,
            ttl_seconds=3600,
        )
        assert mock_urlopen.call_count == 1
        assert len(res1) == 2

        # Second query: hits cache, urlopen is not called again
        res2 = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=False,
            cache_dir=tmp_path,
            ttl_seconds=3600,
        )
        assert mock_urlopen.call_count == 1
        assert res2 == res1

        # Third query with refresh_cache=True: bypasses cache and hits network
        res3 = search_lro_nac_overlap(
            region_bounds=TEST_BOUNDS,
            refresh_cache=True,
            cache_dir=tmp_path,
            ttl_seconds=3600,
        )
        assert mock_urlopen.call_count == 2
        assert len(res3) == 2
