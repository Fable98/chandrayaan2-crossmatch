"""
matches.py — GET /triplets/{triplet_id}/matches

Returns LoFTR match points (post-RANSAC) between OHRC and TMC for a given
triplet, plus a re-derived 3×3 homography matrix.

The backend does NOT run any matching. All pixel data comes from the ML
team's matches.json; geographic coordinates are computed at load time
from the shared TripletBounds (see geo.py and loader.py).
"""

from fastapi import APIRouter, HTTPException

from data import loader
from schemas import MatchesResponse

router = APIRouter(tags=["matches"])


@router.get("/triplets/{triplet_id}/matches", response_model=MatchesResponse)
def get_matches(triplet_id: str):
    """
    Return OHRC ↔ TMC match points and homography for one triplet.

    Each match includes:
    - ohrc_px / tmc_px: pixel coordinates from the ML team's output
    - ohrc_latlon / tmc_latlon: geographic coordinates computed from
      the sensor footprint corners at load time
    - confidence: per-match confidence score from LoFTR

    If no match data exists for this triplet (e.g., ML pipeline hasn't
    processed it yet), returns 200 with an empty matches list — not 404.
    This keeps the frontend robust for partially-processed regions.
    """
    # First verify the triplet exists
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    match_data = loader.get_matches(triplet_id)

    # If no match data exists for this triplet, return empty — not 404
    if match_data is None:
        return MatchesResponse(
            triplet_id=triplet_id,
            num_matches=0,
            homography=None,
            matches=[],
        )

    return MatchesResponse(
        triplet_id=match_data["triplet_id"],
        num_matches=len(match_data["matches"]),
        homography=match_data["homography"],
        metrics=match_data.get("metrics"),
        matches=match_data["matches"],
    )
