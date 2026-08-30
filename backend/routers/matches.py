"""
matches.py — GET /triplets/{triplet_id}/matches

Returns LoFTR match points (post-RANSAC) between OHRC and TMC for a given
triplet, plus the pre-computed 3×3 homography matrix.

The backend does NOT run any matching or transform computation. All data is
produced by the ML team and loaded from disk at startup.
"""

from fastapi import APIRouter, HTTPException

from data import loader
from schemas import MatchesResponse

router = APIRouter(tags=["matches"])


@router.get("/triplets/{triplet_id}/matches", response_model=MatchesResponse)
def get_matches(triplet_id: str):
    """
    Return OHRC ↔ TMC match points and homography for one triplet.

    Each match includes pixel coordinates on both images and a confidence
    score. The homography is the 3×3 matrix the ML team already computed —
    the frontend can use it directly without inversion.
    """
    # First verify the triplet exists
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    match_data = loader.get_matches(triplet_id)
    if match_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No match data available for triplet '{triplet_id}'",
        )

    return MatchesResponse(
        triplet_id=match_data["triplet_id"],
        num_matches=len(match_data["matches"]),
        homography=match_data["homography"],
        matches=match_data["matches"],
    )
