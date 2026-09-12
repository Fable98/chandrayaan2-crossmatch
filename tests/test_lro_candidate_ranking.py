"""
tests/test_lro_candidate_ranking.py — Unit tests for LRO NAC Candidate Ranking (Step 2)
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from lro_ode_client import (
    OVERLAP_WEIGHT,
    INCIDENCE_WEIGHT,
    rank_candidates,
)

TARGET_BOUNDS = {
    "west_lon": 336.40,
    "east_lon": 336.60,
    "south_lat": -3.50,
    "north_lat": -3.30,
}
# Target area = (336.60 - 336.40) * (-3.30 - -3.50) = 0.20 * 0.20 = 0.04


def test_ranking_weights_defined():
    """Verify named constants for ranking weights."""
    assert OVERLAP_WEIGHT == 0.8
    assert INCIDENCE_WEIGHT == 0.2
    assert OVERLAP_WEIGHT + INCIDENCE_WEIGHT == pytest.approx(1.0)


def test_rank_candidates_full_overlap_and_incidence_match():
    """Candidates with full overlap and closer incidence angle rank higher."""
    target_inc = 15.0

    candidates = [
        {
            "product_id": "CAND_POOR_INCIDENCE",
            "label_url": "https://example.com/c1.lbl",
            "download_urls": ["https://example.com/c1.img"],
            # Encloses target entirely -> 100% overlap
            "footprint_bounds": {"west_lon": 336.30, "east_lon": 336.70, "south_lat": -3.60, "north_lat": -3.20},
            "incidence_angle_deg": 60.0,  # delta = 45.0
        },
        {
            "product_id": "CAND_EXCELLENT_INCIDENCE",
            "label_url": "https://example.com/c2.lbl",
            "download_urls": ["https://example.com/c2.img"],
            # Encloses target entirely -> 100% overlap
            "footprint_bounds": {"west_lon": 336.30, "east_lon": 336.70, "south_lat": -3.60, "north_lat": -3.20},
            "incidence_angle_deg": 16.0,  # delta = 1.0
        },
    ]

    ranked = rank_candidates(candidates, TARGET_BOUNDS, target_incidence_angle=target_inc)
    assert len(ranked) == 2
    assert ranked[0]["product_id"] == "CAND_EXCELLENT_INCIDENCE"
    assert ranked[1]["product_id"] == "CAND_POOR_INCIDENCE"

    # Verify provenance tags
    assert ranked[0]["overlap_score_is_derived"] is True
    assert "Geographic intersection-over-target-area" in ranked[0]["overlap_score_derivation"]
    assert ranked[0]["ranking_score_is_derived"] is True
    assert "Weighted composite" in ranked[0]["ranking_score_derivation"]
    assert ranked[0]["overlap_score"] == 1.0


def test_rank_candidates_missing_footprint_ranks_below_measurable_overlap():
    """A candidate with footprint_bounds=None must rank below any candidate with measurable overlap."""
    candidates = [
        {
            "product_id": "NO_FOOTPRINT",
            "label_url": None,
            "download_urls": [],
            "footprint_bounds": None,
            "incidence_angle_deg": 15.0,
        },
        {
            "product_id": "PARTIAL_OVERLAP",
            "label_url": None,
            "download_urls": [],
            # Covers right half of target (west: 336.50 to 336.60, south: -3.50 to -3.30) -> 50% overlap
            "footprint_bounds": {"west_lon": 336.50, "east_lon": 336.70, "south_lat": -3.50, "north_lat": -3.30},
            "incidence_angle_deg": 80.0,
        },
    ]

    ranked = rank_candidates(candidates, TARGET_BOUNDS, target_incidence_angle=15.0)
    assert ranked[0]["product_id"] == "PARTIAL_OVERLAP"
    assert ranked[0]["overlap_score"] == pytest.approx(0.50, abs=0.01)
    assert ranked[1]["product_id"] == "NO_FOOTPRINT"
    assert ranked[1]["overlap_score"] == 0.0


def test_rank_candidates_missing_incidence_ranks_on_overlap_alone():
    """Missing incidence angle ranks on overlap alone without crash or penalty."""
    candidates = [
        {
            "product_id": "CAND_NO_INCIDENCE_HIGH_OVERLAP",
            "label_url": None,
            "download_urls": [],
            # 100% overlap
            "footprint_bounds": {"west_lon": 336.30, "east_lon": 336.70, "south_lat": -3.60, "north_lat": -3.20},
        },
        {
            "product_id": "CAND_WITH_INCIDENCE_LOW_OVERLAP",
            "label_url": None,
            "download_urls": [],
            # 25% overlap
            "footprint_bounds": {"west_lon": 336.40, "east_lon": 336.45, "south_lat": -3.50, "north_lat": -3.30},
            "incidence_angle_deg": 15.0,
        },
    ]

    # With target_incidence_angle=None
    ranked1 = rank_candidates(candidates, TARGET_BOUNDS, target_incidence_angle=None)
    assert ranked1[0]["product_id"] == "CAND_NO_INCIDENCE_HIGH_OVERLAP"
    assert ranked1[0]["overlap_score"] == 1.0
    assert ranked1[0]["ranking_score"] == 1.0
    assert "overlap score alone" in ranked1[0]["ranking_score_derivation"]

    # Even with target_incidence_angle=15.0, the 100% overlap candidate beats 25% overlap
    ranked2 = rank_candidates(candidates, TARGET_BOUNDS, target_incidence_angle=15.0)
    assert ranked2[0]["product_id"] == "CAND_NO_INCIDENCE_HIGH_OVERLAP"
    assert ranked2[0]["ranking_score"] == 1.0


def test_rank_candidates_empty_input():
    """Empty list returns empty list."""
    assert rank_candidates([], TARGET_BOUNDS) == []
