"""
triplets.py — GET /triplets, GET /triplets/{triplet_id}, DELETE /triplets/{triplet_id}

Returns triplet metadata with shared TripletBounds from in-memory cache.
DELETE removes a generated dataset's on-disk bundle and reloads the catalog.
"""

import json
import os
import re
import shutil

from fastapi import APIRouter, Depends, HTTPException

from data import loader
from schemas import TripletListResponse, TripletSummary

try:
    from routers.auth import get_current_user
except ImportError:  # pragma: no cover - direct-router test path
    from backend.routers.auth import get_current_user  # type: ignore

router = APIRouter(tags=["triplets"])

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
# Curated seed regions ship force-tracked in git; deleting them locally
# breaks the vault until `git restore`. Require explicit ?force=true.
_CURATED_RE = re.compile(r"^region_\d+$", re.IGNORECASE)


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


@router.delete("/triplets/{triplet_id}")
def delete_triplet(
    triplet_id: str,
    force: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """Delete a dataset bundle and reload the in-memory catalog.

    Removes ``processed_triplets/<id>/``, matching ``user_triplets.json``
    rows, ``matches/<id>_*.json`` files and ``registration_output/<id>/``,
    then calls ``loader.load_all()``. Generated ids (``region_auto_*``,
    ``triplet_*``) delete freely; curated ``region_NNN`` seeds require
    ``?force=true`` since they are force-tracked in git.
    """
    if not _ID_RE.match(triplet_id) or ".." in triplet_id or "/" in triplet_id:
        raise HTTPException(status_code=400, detail="Invalid triplet id")
    if _CURATED_RE.match(triplet_id) and not force:
        raise HTTPException(
            status_code=400,
            detail=f"'{triplet_id}' is a curated seed region. Retry with ?force=true to delete it anyway.",
        )

    triplets_dir = getattr(loader, "PROCESSED_TRIPLETS_DIR", None)
    data_dir = getattr(loader, "DATA_DIR", None)
    repo_root = getattr(loader, "REPO_ROOT", None)
    removed: list[str] = []

    def _rmtree(path: str) -> None:
        real = os.path.realpath(path)
        # Never delete outside the known data roots.
        roots = [r for r in (triplets_dir, data_dir, repo_root) if r]
        if not any(real == os.path.realpath(str(r)) or real.startswith(os.path.realpath(str(r)) + os.sep) for r in roots):
            return
        if os.path.isdir(real):
            shutil.rmtree(real, ignore_errors=True)
            removed.append(real)

    def _rmfile(path: str) -> None:
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                pass

    if triplets_dir:
        _rmtree(os.path.join(str(triplets_dir), triplet_id))
    if data_dir and os.path.realpath(str(data_dir)) != (os.path.realpath(str(triplets_dir)) if triplets_dir else None):
        _rmtree(os.path.join(str(data_dir), triplet_id))
        # Per-triplet match + transform files.
        _rmfile(os.path.join(str(data_dir), "matches", f"{triplet_id}_matches.json"))
        _rmfile(os.path.join(str(data_dir), "matches", f"{triplet_id}_transform.json"))
    if repo_root:
        _rmtree(os.path.join(str(repo_root), "registration_output", triplet_id))
        # Prune user_triplets.json rows for this region (list or {"triplets": [...]}).
        manifest_path = os.path.join(str(data_dir or ""), "user_triplets.json")
        if manifest_path and os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                changed = False
                if isinstance(manifest, list):
                    kept = [t for t in manifest if not (isinstance(t, dict) and (t.get("region_id") == triplet_id or t.get("triplet_id") == triplet_id))]
                    changed = len(kept) != len(manifest)
                    manifest = kept
                elif isinstance(manifest, dict) and isinstance(manifest.get("triplets"), list):
                    rows = manifest["triplets"]
                    kept = [t for t in rows if not (isinstance(t, dict) and (t.get("region_id") == triplet_id or t.get("triplet_id") == triplet_id))]
                    changed = len(kept) != len(rows)
                    manifest["triplets"] = kept
                if changed:
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2)
                    removed.append(manifest_path + f" (pruned {triplet_id} rows)")
            except (OSError, ValueError):
                pass

    if not removed:
        raise HTTPException(status_code=404, detail=f"No on-disk bundle found for '{triplet_id}'")

    try:
        loader.load_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deleted files but catalog reload failed: {exc}")

    return {"triplet_id": triplet_id, "deleted": True, "removed": removed}

