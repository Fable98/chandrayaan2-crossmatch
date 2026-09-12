"""
ML_model/lro_ode_client.py — ODE REST API Client for LRO NAC Automated Discovery

Queries the Washington University ODE (Orbital Data Explorer) REST API for LRO NAC
(Narrow Angle Camera) products overlapping lunar target bounding boxes.

Uses stdlib urllib only (no third-party HTTP libraries).
Implements defensive response parsing, on-disk caching with TTL/refresh support,
and graceful error handling.

NOTE ON ODE FIELD NAMES:
The ODE REST API field names handled here are defensively matched (.get with fallbacks)
based on published ODE REST API documentation. Because test environments lack external
network access, actual live ODE field names will be verified during first live operation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from pds3_image_decoder import decode_pds3_img_to_array
except ImportError:
    from ML_model.pds3_image_decoder import decode_pds3_img_to_array

try:
    from lro_pds3_parser import read_pds3_label
except ImportError:
    from ML_model.lro_pds3_parser import read_pds3_label

try:
    from data_preprocessing_pipeline.scripts.prepare_lro_nac_pair import prepare_pair_for_region
except ImportError:
    REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))
    try:
        from data_preprocessing_pipeline.scripts.prepare_lro_nac_pair import prepare_pair_for_region
    except ImportError:
        prepare_pair_for_region = None  # type: ignore[assignment]

logger = logging.getLogger("ML_model.lro_ode_client")

REPO_ROOT = Path(__file__).resolve().parent.parent
ODE_REST_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_TIMEOUT_S = 30
CACHE_DIR = REPO_ROOT / ".cache" / "lro_ode"
CACHE_TTL_S = 86400  # 24 hours

OVERLAP_WEIGHT = 0.8
INCIDENCE_WEIGHT = 0.2

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard download size guard
MIN_OVERLAP_THRESHOLD = 0.20  # Minimum 20% footprint overlap required
DOWNLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunk


def _get_cache_path(bounds: dict, product_type: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Generates a stable cache filename for a given bounding box and product type."""
    cache_key = (
        f"{float(bounds.get('west_lon', 0)):.6f}_"
        f"{float(bounds.get('east_lon', 0)):.6f}_"
        f"{float(bounds.get('south_lat', 0)):.6f}_"
        f"{float(bounds.get('north_lat', 0)):.6f}_"
        f"{product_type.upper()}"
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"ode_query_{digest}.json"


def _get_float_defensive(d: dict, keys: list[str]) -> Optional[float]:
    """Defensively attempts to find and convert any of the given keys to float."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
    return None


def _extract_product_list(raw_json: Any) -> List[dict]:
    """
    Defensively extracts candidate product dictionaries from various possible
    ODE REST JSON response shapes.
    """
    if isinstance(raw_json, list):
        return [p for p in raw_json if isinstance(p, dict)]

    if not isinstance(raw_json, dict):
        return []

    # Case 1: Standard ODEResults -> Products -> Product
    ode_results = raw_json.get("ODEResults")
    if isinstance(ode_results, dict):
        products_container = ode_results.get("Products")
        if isinstance(products_container, dict):
            prods = products_container.get("Product")
            if isinstance(prods, list):
                return [p for p in prods if isinstance(p, dict)]
            elif isinstance(prods, dict):
                return [prods]
        elif isinstance(products_container, list):
            return [p for p in products_container if isinstance(p, dict)]

    # Case 2: Top-level Products / Product / results
    for key in ("Products", "products", "Product", "product", "results", "Items"):
        val = raw_json.get(key)
        if isinstance(val, list):
            return [p for p in val if isinstance(p, dict)]
        elif isinstance(val, dict):
            sub = val.get("Product") or val.get("product")
            if isinstance(sub, list):
                return [p for p in sub if isinstance(p, dict)]
            elif isinstance(sub, dict):
                return [sub]
            return [val]

    return []


def _parse_candidate_product(prod: dict) -> dict:
    """
    Parses a single product entry into the strict candidate contract.

    Mandatory keys:
      - product_id: str
      - label_url: Optional[str]
      - download_urls: list[str]
      - footprint_bounds: Optional[dict] with west_lon, east_lon, south_lat, north_lat

    Optional keys:
      - incidence_angle_deg, emission_angle_deg, phase_angle_deg
    """
    # 1. Product ID
    pid: Optional[str] = None
    for k in ("pdsid", "PDSID", "product_id", "ProductId", "Product_id", "id", "Product"):
        val = prod.get(k)
        if isinstance(val, str) and val.strip():
            pid = val.strip()
            break
    if not pid:
        pid = "UNKNOWN_PRODUCT"

    # 2. File URLs (Label and Data)
    label_url: Optional[str] = prod.get("LabelURL") or prod.get("label_url") or prod.get("LabelUrl")
    download_urls: List[str] = []

    files_obj = (
        prod.get("Product_files")
        or prod.get("Product_Files")
        or prod.get("Files")
        or prod.get("files")
    )
    file_list: List[Any] = []
    if isinstance(files_obj, list):
        file_list = files_obj
    elif isinstance(files_obj, dict):
        sub_f = (
            files_obj.get("Product_file")
            or files_obj.get("Product_File")
            or files_obj.get("File")
            or files_obj.get("file")
        )
        if isinstance(sub_f, list):
            file_list = sub_f
        elif isinstance(sub_f, dict):
            file_list = [sub_f]

    for f in file_list:
        if isinstance(f, dict):
            url = f.get("URL") or f.get("url") or f.get("DownloadURL") or f.get("ProductURL")
            ftype = str(f.get("Type", "") or f.get("type", "") or f.get("Description", "")).lower()
            if url and isinstance(url, str):
                download_urls.append(url)
                if (url.lower().endswith(".lbl") or "label" in ftype) and not label_url:
                    label_url = url
        elif isinstance(f, str):
            download_urls.append(f)
            if f.lower().endswith(".lbl") and not label_url:
                label_url = f

    # Check top-level URL fields if download_urls is still empty
    if not download_urls:
        for k in ("ProductURL", "product_url", "URL", "url", "download_url"):
            u = prod.get(k)
            if isinstance(u, str) and u.strip():
                download_urls.append(u.strip())
                if u.lower().endswith(".lbl") and not label_url:
                    label_url = u.strip()

    # 3. Footprint bounds
    w = _get_float_defensive(
        prod,
        [
            "Westernmost_longitude",
            "westernmost_longitude",
            "west_lon",
            "westlon",
            "WesternmostLongitude",
            "Min_longitude",
            "min_lon",
        ],
    )
    e = _get_float_defensive(
        prod,
        [
            "Easternmost_longitude",
            "easternmost_longitude",
            "east_lon",
            "eastlon",
            "EasternmostLongitude",
            "Max_longitude",
            "max_lon",
        ],
    )
    s = _get_float_defensive(
        prod,
        [
            "Minimum_latitude",
            "minimum_latitude",
            "south_lat",
            "minlat",
            "MinimumLatitude",
            "Min_latitude",
            "min_lat",
        ],
    )
    n = _get_float_defensive(
        prod,
        [
            "Maximum_latitude",
            "maximum_latitude",
            "north_lat",
            "maxlat",
            "MaximumLatitude",
            "Max_latitude",
            "max_lat",
        ],
    )

    footprint_bounds: Optional[Dict[str, float]] = None
    if all(v is not None for v in (w, e, s, n)):
        footprint_bounds = {
            "west_lon": float(w),  # type: ignore[arg-type]
            "east_lon": float(e),  # type: ignore[arg-type]
            "south_lat": float(s),  # type: ignore[arg-type]
            "north_lat": float(n),  # type: ignore[arg-type]
        }

    candidate: Dict[str, Any] = {
        "product_id": pid,
        "label_url": label_url,
        "download_urls": download_urls,
        "footprint_bounds": footprint_bounds,
    }

    # 4. Optional observation geometry fields
    inc = _get_float_defensive(
        prod,
        [
            "Incidence_angle",
            "incidence_angle",
            "IncidenceAngle",
            "incidence_angle_deg",
            "incidence",
            "Center_incidence_angle",
        ],
    )
    if inc is not None:
        candidate["incidence_angle_deg"] = inc

    em = _get_float_defensive(
        prod,
        [
            "Emission_angle",
            "emission_angle",
            "EmissionAngle",
            "emission_angle_deg",
            "emission",
            "Center_emission_angle",
        ],
    )
    if em is not None:
        candidate["emission_angle_deg"] = em

    ph = _get_float_defensive(
        prod,
        [
            "Phase_angle",
            "phase_angle",
            "PhaseAngle",
            "phase_angle_deg",
            "phase",
            "Center_phase_angle",
        ],
    )
    if ph is not None:
        candidate["phase_angle_deg"] = ph

    return candidate


def search_lro_nac_overlap(
    region_bounds: Dict[str, float],
    product_type: str = "EDRNAC",
    refresh_cache: bool = False,
    cache_dir: Path = CACHE_DIR,
    ttl_seconds: int = CACHE_TTL_S,
) -> List[Dict[str, Any]]:
    """
    Queries Washington University ODE REST API for LRO NAC products overlapping
    the specified geographic bounding box.

    Parameters:
        region_bounds: Dict with keys 'west_lon', 'east_lon', 'south_lat', 'north_lat'.
        product_type: ODE product type (default 'EDRNAC').
        refresh_cache: If True, bypasses on-disk cache and makes a fresh network query.
        cache_dir: Directory for on-disk query cache.
        ttl_seconds: Cache time-to-live in seconds (default 24h).

    Returns:
        List of candidate dictionaries, each strictly containing:
            - 'product_id': str
            - 'label_url': str | None
            - 'download_urls': list[str]
            - 'footprint_bounds': dict | None
            And optional geometry keys (e.g. 'incidence_angle_deg').
    """
    required_keys = {"west_lon", "east_lon", "south_lat", "north_lat"}
    if not required_keys.issubset(region_bounds.keys()):
        logger.warning(
            "search_lro_nac_overlap received invalid bounds dict: missing %s",
            required_keys - set(region_bounds.keys()),
        )
        return []

    # Check cache unless refresh_cache requested
    cache_path = _get_cache_path(region_bounds, product_type, cache_dir)
    if not refresh_cache and cache_path.exists():
        try:
            mtime = cache_path.stat().st_mtime
            if (time.time() - mtime) < ttl_seconds:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list):
                    logger.info("Loaded %d ODE candidates from cache: %s", len(cached_data), cache_path.name)
                    return cached_data
        except Exception as exc:
            logger.warning("Failed to read ODE cache at %s: %s", cache_path, exc)

    # Build ODE REST query parameters
    params = {
        "query": "product",
        "results": "f",
        "output": "JSON",
        "target": "moon",
        "ihid": "lro",
        "iid": "lroc",
        "pt": product_type,
        "minlat": f"{float(region_bounds['south_lat']):.6f}",
        "maxlat": f"{float(region_bounds['north_lat']):.6f}",
        "westernlon": f"{float(region_bounds['west_lon']):.6f}",
        "easternlon": f"{float(region_bounds['east_lon']):.6f}",
    }

    url = f"{ODE_REST_BASE_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Querying ODE REST: %s", url)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Chandrayaan2-Crossmatch-Discovery/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=ODE_TIMEOUT_S) as resp:
            content_bytes = resp.read()
            raw_text = content_bytes.decode("utf-8", errors="ignore")
            raw_json = json.loads(raw_text)
    except urllib.error.HTTPError as exc:
        logger.warning("ODE REST HTTP error %d: %s", exc.code, exc.reason)
        return []
    except urllib.error.URLError as exc:
        logger.warning("ODE REST connection/URL error: %s", exc.reason)
        return []
    except (socket.timeout, TimeoutError) as exc:
        logger.warning("ODE REST request timed out after %ds: %s", ODE_TIMEOUT_S, exc)
        return []
    except json.JSONDecodeError as exc:
        logger.warning("ODE REST returned non-JSON response: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Unexpected error querying ODE REST: %s", exc)
        return []

    # Parse candidates defensively
    products = _extract_product_list(raw_json)
    candidates = [_parse_candidate_product(p) for p in products]

    # Save to on-disk cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)
    except Exception as exc:
        logger.warning("Could not write to ODE cache at %s: %s", cache_path, exc)

    logger.info("Found %d LRO NAC candidates overlapping bounds via ODE", len(candidates))
    return candidates


def rank_candidates(
    candidates: List[Dict[str, Any]],
    target_bounds: Dict[str, float],
    target_incidence_angle: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Ranks LRO NAC candidate products based on geographic overlap with target bounds
    and solar illumination incidence angle similarity.

    Parameters:
        candidates: List of candidate dicts conforming to search_lro_nac_overlap contract.
        target_bounds: Target region bounds (west_lon, east_lon, south_lat, north_lat).
        target_incidence_angle: Optional target incidence angle in degrees.

    Returns:
        Sorted list of candidate dicts with added ranking and provenance metadata:
            - 'overlap_score': float [0.0, 1.0]
            - 'overlap_score_is_derived': True
            - 'overlap_score_derivation': str
            - 'ranking_score': float [0.0, 1.0]
            - 'ranking_score_is_derived': True
            - 'ranking_score_derivation': str
    """
    if not candidates:
        return []

    tw = float(target_bounds.get("west_lon", 0.0))
    te = float(target_bounds.get("east_lon", 0.0))
    ts = float(target_bounds.get("south_lat", 0.0))
    tn = float(target_bounds.get("north_lat", 0.0))

    target_area = max(0.0, (te - tw)) * max(0.0, (tn - ts))

    scored: List[Dict[str, Any]] = []

    for c in candidates:
        entry = dict(c)
        fp = c.get("footprint_bounds")

        if fp is None or target_area <= 0.0:
            overlap_ratio = 0.0
        else:
            cw = float(fp.get("west_lon", 0.0))
            ce = float(fp.get("east_lon", 0.0))
            cs = float(fp.get("south_lat", 0.0))
            cn = float(fp.get("north_lat", 0.0))

            inter_w = max(tw, cw)
            inter_e = min(te, ce)
            inter_s = max(ts, cs)
            inter_n = min(tn, cn)

            if inter_e > inter_w and inter_n > inter_s:
                inter_area = (inter_e - inter_w) * (inter_n - inter_s)
                overlap_ratio = min(1.0, max(0.0, inter_area / target_area))
            else:
                overlap_ratio = 0.0

        # Provenance-tagged overlap score
        entry["overlap_score"] = round(float(overlap_ratio), 4)
        entry["overlap_score_is_derived"] = True
        entry["overlap_score_derivation"] = (
            "Geographic intersection-over-target-area ratio between candidate footprint and target region bounds."
        )

        cand_inc = c.get("incidence_angle_deg")
        if target_incidence_angle is not None and cand_inc is not None:
            inc_delta = abs(float(cand_inc) - float(target_incidence_angle))
            # Illumination similarity in [0, 1] normalized against 90 deg max delta
            inc_sim = max(0.0, 1.0 - (inc_delta / 90.0))
            ranking_score = (OVERLAP_WEIGHT * overlap_ratio) + (INCIDENCE_WEIGHT * inc_sim)
            derivation = (
                f"Weighted composite: {OVERLAP_WEIGHT}*overlap_score + {INCIDENCE_WEIGHT}*incidence_similarity "
                f"(delta={inc_delta:.2f} deg)."
            )
            entry["incidence_delta_deg"] = round(float(inc_delta), 2)
        else:
            ranking_score = overlap_ratio
            derivation = (
                "Ranked on overlap score alone (target or candidate illumination incidence angle not available)."
            )

        entry["ranking_score"] = round(float(ranking_score), 4)
        entry["ranking_score_is_derived"] = True
        entry["ranking_score_derivation"] = derivation

        scored.append(entry)

    # Sort descending by ranking score. Candidates without footprint (overlap=0) rank below any candidate with overlap > 0.
    scored.sort(
        key=lambda x: (
            x["ranking_score"],
            x["overlap_score"],
            -(x.get("incidence_delta_deg") if x.get("incidence_delta_deg") is not None else 999.0),
        ),
        reverse=True,
    )

    return scored


def _stream_download(url: str, dest_path: Path, max_bytes: int = MAX_DOWNLOAD_BYTES) -> int:
    """
    Downloads URL content to dest_path using chunked streaming.
    Enforces unconditional hard size guard of max_bytes.
    Deletes any partial file on error or size violation.
    """
    downloaded = 0
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Chandrayaan2-Crossmatch-Downloader/1.0"},
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(req, timeout=ODE_TIMEOUT_S) as resp, open(dest_path, "wb") as f_out:
            while True:
                chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(
                        f"Download exceeded unconditional size limit: {downloaded} > {max_bytes} bytes"
                    )
                f_out.write(chunk)
    except Exception as exc:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        raise exc

    return downloaded


def fetch_and_prepare_lro_nac(
    region_bounds: Dict[str, float],
    region_id: str,
    output_dir: Path,
    incidence_angle: Optional[float] = None,
    product_type: str = "EDRNAC",
    refresh_cache: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Discovers, downloads, decodes, and stages an overlapping LRO NAC frame for a given region.

    Parameters:
        region_bounds: Target region bounds (west_lon, east_lon, south_lat, north_lat).
        region_id: Region ID (e.g. 'region_001').
        output_dir: Exact destination directory (e.g. data_preprocessing_pipeline/lro_nac_real/<region_id>/).
        incidence_angle: Optional target solar illumination incidence angle in degrees.
        product_type: ODE product type (default 'EDRNAC').
        refresh_cache: If True, refreshes ODE query cache.

    Returns:
        Manifest dictionary from prepare_pair_for_region, or None on failure/unmet threshold.
    """
    # 1. PREREQUISITE CHECK FIRST
    triplet_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / region_id
    triplet_manifest = triplet_dir / "manifest.json"
    triplet_ohrc = triplet_dir / "ohrc_512.png"

    if not triplet_manifest.is_file() or not triplet_ohrc.is_file():
        logger.error(
            "Prerequisite check failed: region_id '%s' not yet ingested into processed_triplets "
            "(requires manifest.json and ohrc_512.png under %s)",
            region_id,
            triplet_dir,
        )
        return None

    # 2. Search & rank candidates
    candidates = search_lro_nac_overlap(
        region_bounds=region_bounds,
        product_type=product_type,
        refresh_cache=refresh_cache,
    )
    if not candidates:
        logger.warning("No LRO NAC candidates found overlapping region '%s'", region_id)
        return None

    ranked = rank_candidates(
        candidates=candidates,
        target_bounds=region_bounds,
        target_incidence_angle=incidence_angle,
    )
    if not ranked:
        logger.warning("No rankable LRO NAC candidates for region '%s'", region_id)
        return None

    top = ranked[0]
    overlap_score = float(top.get("overlap_score", 0.0))
    if overlap_score < MIN_OVERLAP_THRESHOLD:
        logger.warning(
            "Top candidate %s overlap score (%.2f%%) failed minimum overlap threshold (%.2f%%) for region '%s'",
            top.get("product_id"),
            overlap_score * 100.0,
            MIN_OVERLAP_THRESHOLD * 100.0,
            region_id,
        )
        return None

    # 3. Resolve URLs
    label_url = top.get("label_url")
    download_urls = top.get("download_urls", [])

    img_url = None
    for u in download_urls:
        u_lower = u.lower()
        if u_lower.endswith(".img") or u_lower.endswith(".dat"):
            img_url = u
            break
        if not label_url and u_lower.endswith(".lbl"):
            label_url = u

    if not label_url or not img_url:
        logger.error(
            "Top candidate %s missing required download URLs (label=%s, img=%s)",
            top.get("product_id"),
            label_url,
            img_url,
        )
        return None

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 4. Snapshot existing files in output_dir before any operations
    pre_existing_paths = set(out_dir_path.rglob("*")) if out_dir_path.exists() else set()

    temp_files_to_clean: List[Path] = []
    try:
        t_start = time.time()

        # Temporary label file
        temp_lbl = tempfile.NamedTemporaryFile(suffix=".LBL", delete=False)
        temp_lbl.close()
        temp_lbl_path = Path(temp_lbl.name)
        temp_files_to_clean.append(temp_lbl_path)

        lbl_bytes = _stream_download(label_url, temp_lbl_path, max_bytes=10 * 1024 * 1024)

        # Validate downloaded label immediately with ML_model/lro_pds3_parser.py
        try:
            parsed_lbl = read_pds3_label(temp_lbl_path)
            if not parsed_lbl:
                raise ValueError("Parsed label dictionary is empty")
        except Exception as lbl_exc:
            logger.error("Downloaded PDS3 label validation failed for %s: %s", top.get("product_id"), lbl_exc)
            raise lbl_exc

        # Temporary image file
        temp_img = tempfile.NamedTemporaryFile(suffix=".IMG", delete=False)
        temp_img.close()
        temp_img_path = Path(temp_img.name)
        temp_files_to_clean.append(temp_img_path)

        img_bytes = _stream_download(img_url, temp_img_path, max_bytes=MAX_DOWNLOAD_BYTES)
        download_duration_s = time.time() - t_start

        # 5. Decode .IMG to numpy array
        raw_array = decode_pds3_img_to_array(temp_img_path, parsed_lbl)

        # 6. Write decoded array to temporary PNG file
        temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_png.close()
        temp_png_path = Path(temp_png.name)
        temp_files_to_clean.append(temp_png_path)

        if raw_array.dtype == np.uint8:
            png_data = raw_array
        else:
            png_data = cv2.normalize(raw_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(str(temp_png_path), png_data)

        # 7. Call prepare_pair_for_region with temporary file paths
        pair_manifest = prepare_pair_for_region(
            region_id=region_id,
            raw_nac_image=temp_png_path,
            raw_nac_label=temp_lbl_path,
            output_dir=out_dir_path,
        )

        # 8. MANDATORY Anti-fabrication check
        provenance = pair_manifest.get("reference_provenance")
        if provenance != "real_downloaded_cdr":
            raise RuntimeError(
                f"Anti-fabrication check failed for region '{region_id}': prepare_pair_for_region fell back to "
                f"reference_provenance='{provenance}' instead of 'real_downloaded_cdr'. Synthetic references "
                f"must never be passed off as real."
            )

        logger.info(
            "Selected candidate %s for region %s | Overlap: %.2f%% | Inc Delta: %s deg | "
            "File Size: %.2f MB | Download Duration: %.2fs",
            top.get("product_id"),
            region_id,
            overlap_score * 100.0,
            f"{top.get('incidence_delta_deg'):.2f}" if top.get("incidence_delta_deg") is not None else "N/A",
            (lbl_bytes + img_bytes) / (1024 * 1024),
            download_duration_s,
        )

        return pair_manifest

    except Exception as exc:
        logger.error("fetch_and_prepare_lro_nac failed for region '%s': %s", region_id, exc)
        # Safe cleanup: Delete ONLY newly created paths under output_dir, preserve pre-existing files
        current_paths = set(out_dir_path.rglob("*")) if out_dir_path.exists() else set()
        new_paths = current_paths - pre_existing_paths
        for p in sorted(new_paths, reverse=True):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            except Exception as cl_exc:
                logger.warning("Safe cleanup error for %s: %s", p, cl_exc)
        raise exc

    finally:
        # Clean up temporary files
        for p in temp_files_to_clean:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


