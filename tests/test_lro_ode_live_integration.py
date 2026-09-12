"""
tests/test_lro_ode_live_integration.py — Live ODE REST Integration Test (Step 5)

This integration test queries the live Washington University ODE REST API to verify
live field-name parsing against actual LROC EDR/CDR products for Chandrayaan-2 regions.

Gated behind a skip condition because the test environment lacks external network access.
Live validation must be run in an environment with outbound internet connectivity to
https://oderest.rsl.wustl.edu/.
"""

import os
import socket
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from lro_ode_client import search_lro_nac_overlap, rank_candidates


def _is_network_available() -> bool:
    """Checks if oderest.rsl.wustl.edu is reachable."""
    if os.environ.get("LRO_ODE_LIVE_TEST") != "1":
        return False
    try:
        socket.create_connection(("oderest.rsl.wustl.edu", 443), timeout=3.0)
        return True
    except (socket.timeout, OSError):
        return False


@pytest.mark.skipif(
    not _is_network_available(),
    reason=(
        "Live ODE REST integration test skipped: No external network access in this environment. "
        "ODE field names (Step 1) and PDS3 SAMPLE_TYPE/byte-order assumptions (Step 3a) remain "
        "defensively implemented and unverified against real ODE servers until run with network "
        "access (set LRO_ODE_LIVE_TEST=1 in an environment with outbound HTTPS)."
    ),
)
def test_live_ode_search_region_001():
    """Live test querying ODE for region_001 bounds."""
    region_001_bounds = {
        "west_lon": 336.484646,
        "east_lon": 336.589455,
        "south_lat": -3.518776,
        "north_lat": -3.424168,
    }

    candidates = search_lro_nac_overlap(
        region_bounds=region_001_bounds,
        product_type="EDRNAC",
        refresh_cache=True,
    )

    assert len(candidates) > 0, "Expected at least one overlapping LRO NAC candidate for region_001"
    top_cand = candidates[0]
    assert "product_id" in top_cand
    assert "label_url" in top_cand
    assert "download_urls" in top_cand
    assert "footprint_bounds" in top_cand

    ranked = rank_candidates(candidates, region_001_bounds, target_incidence_angle=5.82)
    assert len(ranked) > 0
    assert ranked[0]["overlap_score"] > 0.0
