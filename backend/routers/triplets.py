"""
triplets.py — GET /triplets and GET /triplets/{triplet_id}

Returns triplet metadata with shared TripletBounds from in-memory cache.
"""

from fastapi import APIRouter, HTTPException

from data import loader
from schemas import TripletListResponse, TripletSummary

router = APIRouter(tags=["triplets"])


@router.get("/triplets", response_model=TripletListResponse)
def list_triplets():
    """Return all available triplets with sensor metadata and intersection footprints."""
    return TripletListResponse(triplets=loader.get_triplets())


@router.get("/triplets/{triplet_id}", response_model=TripletSummary)
def get_triplet(triplet_id: str):
    """Return a single triplet by ID."""
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")
    return TripletSummary(**triplet)


@router.get("/triplets/{triplet_id}/lro-candidates")
def get_lro_candidates(triplet_id: str):
    """Query ODE REST for overlapping LRO NAC candidate frames for this triplet."""
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")
    bounds = triplet.get("bounds")
    if not bounds:
        raise HTTPException(status_code=400, detail="Triplet has no geographic bounds")
    try:
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root / "ML_model") not in sys.path:
            sys.path.insert(0, str(repo_root / "ML_model"))
        from lro_ode_client import search_lro_nac_overlap, rank_candidates
        candidates = search_lro_nac_overlap(bounds)
        inc = triplet.get("ohrc_incidence_angle_deg")
        ranked = rank_candidates(candidates, bounds, target_incidence_angle=inc)
        return {
            "triplet_id": triplet_id,
            "candidates": ranked,
            "bounds": bounds,
            "target_incidence_angle": inc,
        }
    except Exception as exc:
        return {"triplet_id": triplet_id, "candidates": [], "error": str(exc)}

