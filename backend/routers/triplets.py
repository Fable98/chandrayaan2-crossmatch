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
