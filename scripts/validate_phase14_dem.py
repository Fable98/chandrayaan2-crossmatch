"""Phase 14 validation driver: SLDEM2015 ingestion validation (diagnostic only).

Usage:
    python3 scripts/validate_phase14_dem.py [--dem-dir DIR] [--manifest PATH]
        [--solar-json PATH] [--report PATH] [--provenance-dir DIR]
        [--previews-dir DIR] [--repo-root PATH]

Behaviour:
* Resolves the operator DEM directory (CLI > $CH2X_DEM_DIR > data/external/dem/).
* For each contracted tile: presence -> filename/type -> SHA-256 -> raster
  metadata -> coverage of its regions -> per-region crop -> crop statistics.
* Region_001 hillshade preview uses ONLY the historical repository elevations
  (labeled historical_repository_metadata); regions 002-006 require
  operator-supplied authoritative solar metadata (--solar-json) or are refused
  with SOLAR_ELEVATION_REQUIRED.
* Writes evaluation/phase14_dem_ingestion_validation.md with decision exactly
  one of DEM_INGESTION_VALIDATED / DEM_INGESTION_BLOCKED / DEM_INGESTION_FAILED.
* Never performs correspondence matching; never touches production files.

Exit codes: 0 when the report is written (any decision); 2 when report writing
itself fails. The decision is carried by the report, not the exit code, so a
BLOCKED outcome (missing operator files) is not a script crash.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from dem_ingestion import (  # noqa: E402
    ALL_REGION_IDS,
    CODE_VERSION,
    DATASET_ID,
    DEMCoverageError,
    DEMIngestionError,
    DEMMissingError,
    PDS4Metadata,
    REGION_001_HISTORICAL_SOLAR,
    SOLAR_ELEVATION_REQUIRED,
    SolarElevationRequiredError,
    build_provenance_record,
    check_filename_and_type,
    compute_sha256,
    crop_region_window,
    expected_sha256_for_tile,
    find_supplied_file,
    load_dem_manifest,
    load_region_bounds,
    preview_hillshade,
    resolve_dem_dir,
    validate_crop_statistics,
    validate_raster_metadata,
    verify_region_coverage,
    write_provenance_record,
)

PRODUCTION_GUARD_PATHS = [
    "ML_model/matcher_cfog.py",
    "ML_model/ai_verifier.py",
    "ML_model/train_ai_verifier.py",
    "ML_model/ground_truth_matches.json",
    "ML_model/ai_verifier_model.pkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 14 SLDEM2015 ingestion validation")
    parser.add_argument("--dem-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--solar-json", default=None,
                        help="Optional operator-supplied authoritative solar metadata JSON")
    parser.add_argument("--report", default=None)
    parser.add_argument("--provenance-dir", default=None)
    parser.add_argument("--previews-dir", default=None)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser.parse_args()


def load_operator_solar(solar_json: str | None) -> dict:
    """Operator-supplied authoritative solar metadata (PDS4-derived)."""
    if not solar_json:
        return {}
    path = Path(solar_json)
    if not path.is_file():
        raise DEMMissingError(f"Operator solar JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DEMIngestionError(f"Operator solar JSON unreadable: {exc}")
    if not isinstance(data, dict):
        raise DEMIngestionError("Operator solar JSON must map region_id -> metadata object.")
    return data


def solar_status_for_region(region_id: str, region_manifests: dict,
                            operator_solar: dict) -> dict:
    """Resolve solar metadata status; elevation stays unset without authority."""
    manifest_box = region_manifests.get(region_id, {})
    azimuth_ohrc = manifest_box.get("ohrc_sun_azimuth_deg")
    azimuth_tmc2 = manifest_box.get("tmc2_sun_azimuth_deg")
    if region_id == "region_001":
        return {
            "ohrc_azimuth": REGION_001_HISTORICAL_SOLAR["ohrc_sun_azimuth_deg"],
            "tmc2_azimuth": REGION_001_HISTORICAL_SOLAR["tmc2_sun_azimuth_deg"],
            "ohrc_elevation": REGION_001_HISTORICAL_SOLAR["ohrc_sun_elevation_deg"],
            "tmc2_elevation": REGION_001_HISTORICAL_SOLAR["tmc2_sun_elevation_deg"],
            "elevation_provenance": "historical_repository_metadata",
            "renderable": True,
        }
    if region_id in operator_solar:
        entry = operator_solar[region_id] or {}
        pds4 = PDS4Metadata(
            source_image_id=str(entry.get("source_image_id", region_id)),
            pds4_label_path=entry.get("pds4_label_path"),
            observation_time_utc=entry.get("observation_time_utc"),
            solar_azimuth_deg=entry.get("solar_azimuth_deg"),
            solar_elevation_deg=entry.get("solar_elevation_deg"),
            spacecraft_position=entry.get("spacecraft_position"),
            target_body=entry.get("target_body", "Moon"),
            footprint=entry.get("footprint"),
            provenance=str(entry.get("provenance", "operator_supplied_pds4")),
        )
        if pds4.has_authoritative_elevation:
            return {
                "ohrc_azimuth": azimuth_ohrc,
                "tmc2_azimuth": azimuth_tmc2,
                "ohrc_elevation": None,
                "tmc2_elevation": float(pds4.solar_elevation_deg),
                "elevation_provenance": pds4.provenance,
                "renderable": True,
            }
    return {
        "ohrc_azimuth": azimuth_ohrc,
        "tmc2_azimuth": azimuth_tmc2,
        "ohrc_elevation": None,
        "tmc2_elevation": None,
        "elevation_provenance": "unset_pending_authoritative_pds4_label",
        "renderable": False,
    }


def production_integrity(repo_root: Path) -> dict:
    results = {}
    for rel in PRODUCTION_GUARD_PATHS:
        try:
            proc = subprocess.run(
                ["git", "diff", "--", rel],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            diff = proc.stdout.strip()
            results[rel] = {"modified": bool(diff), "diff_bytes": len(diff)}
        except Exception as exc:  # pragma: no cover - environment fallback
            results[rel] = {"modified": None, "error": str(exc)}
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        results["_git_status_short"] = proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        results["_git_status_short"] = f"unavailable: {exc}"
    return results


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    dem_dir = resolve_dem_dir(args.dem_dir, repo_root)
    manifest_path = Path(args.manifest) if args.manifest else (
        repo_root / "data" / "external" / "dem" / "dem_manifest.json"
    )
    report_path = Path(args.report) if args.report else (
        repo_root / "evaluation" / "phase14_dem_ingestion_validation.md"
    )
    provenance_dir = Path(args.provenance_dir) if args.provenance_dir else (
        repo_root / "evaluation" / "phase14_provenance"
    )
    previews_dir = Path(args.previews_dir) if args.previews_dir else (
        repo_root / "evaluation_output" / "phase14_previews"
    )

    manifest = load_dem_manifest(manifest_path)
    tiles = manifest.get("tiles", [])
    region_bounds = load_region_bounds(repo_root)

    # Read-only manifest azimuths for the solar-status table.
    region_manifests = {}
    for rid in ALL_REGION_IDS:
        mp = repo_root / "data_preprocessing_pipeline" / "processed_triplets" / rid / "manifest.json"
        try:
            region_manifests[rid] = json.loads(mp.read_text(encoding="utf-8"))
        except OSError:
            region_manifests[rid] = {}
    operator_solar = load_operator_solar(args.solar_json)

    tile_results: dict[str, dict] = {}
    crop_records: dict[str, dict] = {}
    provenance_paths: dict[str, str] = {}
    hard_failure: str | None = None

    for tile in tiles:
        tile_id = tile.get("tile_id", "unknown_tile")
        entry: dict = {
            "tile_id": tile_id,
            "expected_filename": tile.get("expected_filename"),
            "found_path": None,
            "checksum": None,
            "checksum_status": "NOT_CHECKED",
            "raster": None,
            "raster_error": None,
            "coverage": {},
            "crops": {},
        }
        tile_results[tile_id] = entry
        supplied = find_supplied_file(dem_dir, tile)
        if supplied is None:
            entry["checksum_status"] = "FILE_MISSING"
            continue
        entry["found_path"] = str(supplied)
        try:
            check_filename_and_type(supplied, tile)
        except DEMIngestionError as exc:
            entry["checksum_status"] = "REJECTED"
            entry["raster_error"] = str(exc)
            hard_failure = hard_failure or str(exc)
            continue
        expected = expected_sha256_for_tile(tile)
        if expected is None:
            entry["checksum_status"] = "EXPECTED_CHECKSUM_NOT_SUPPLIED"
            try:
                entry["checksum"] = compute_sha256(supplied)
            except DEMIngestionError as exc:
                entry["raster_error"] = str(exc)
            continue
        try:
            entry["checksum"] = compute_sha256(supplied)
            if entry["checksum"].lower() != expected.lower():
                raise ValueError("mismatch")
            entry["checksum_status"] = "VERIFIED"
        except ValueError:
            entry["checksum_status"] = "MISMATCH"
            hard_failure = (
                f"CHECKSUM_MISMATCH for {supplied.name}: operator checksum does not match file."
            )
            continue
        except DEMIngestionError as exc:
            entry["checksum_status"] = "READ_ERROR"
            entry["raster_error"] = str(exc)
            hard_failure = hard_failure or str(exc)
            continue
        try:
            report = validate_raster_metadata(supplied)
            entry["raster"] = report.to_dict()
        except DEMIngestionError as exc:
            entry["raster_error"] = str(exc)
            hard_failure = hard_failure or str(exc)
            continue
        dem_bounds = (
            report.bounds_west,
            report.bounds_south,
            report.bounds_east,
            report.bounds_north,
        )
        for rid in tile.get("covers_regions", []):
            cov = verify_region_coverage(
                dem_bounds, region_bounds[rid], report.res_x_deg, report.res_y_deg
            )
            cov_record = {
                "requested_bounds": cov["requested_bounds"],
                "dem_tile": tile_id,
                "dem_bounds": cov["dem_bounds"],
                "containment": cov["contained"],
                "margin_pixels": cov["margin_pixels"],
            }
            entry["coverage"][rid] = cov_record
            if not cov["contained"]:
                hard_failure = hard_failure or (
                    f"INSUFFICIENT_COVERAGE: {rid} not contained in {tile_id}."
                )
                continue
            try:
                crop = crop_region_window(
                    supplied, region_bounds[rid], rid, tile_id, entry["checksum"]
                )
                stats = validate_crop_statistics(crop.elevation_m, crop.nodata)
                entry["crops"][rid] = {"stats": stats, "crop_bounds": list(crop.crop_bounds)}
                crop_records[rid] = {"crop": crop, "report": report, "stats": stats}
                record = build_provenance_record(crop, report, stats)
                out = write_provenance_record(record, provenance_dir)
                try:
                    provenance_paths[rid] = str(out.relative_to(repo_root))
                except ValueError:
                    provenance_paths[rid] = str(out)
            except DEMIngestionError as exc:
                entry["crops"][rid] = {"error": str(exc)}
                hard_failure = hard_failure or str(exc)

    # --- Hillshade diagnostics (read-only; never guessed) ---
    hillshade_status: dict[str, dict] = {}
    previews_dir.mkdir(parents=True, exist_ok=True)
    for rid in ALL_REGION_IDS:
        solar = solar_status_for_region(rid, region_manifests, operator_solar)
        status: dict = {"solar": solar, "rendered": False, "preview_path": None, "note": ""}
        if rid in crop_records and solar["renderable"]:
            which = "ohrc_elevation" if rid == "region_001" else "tmc2_elevation"
            elev = solar.get(which)
            azim = solar.get("ohrc_azimuth") if rid == "region_001" else solar.get("tmc2_azimuth")
            # Region_001 diagnostic renders both historical geometries.
            geoms = (
                [("ohrc", solar["ohrc_azimuth"], solar["ohrc_elevation"]),
                 ("tmc2", solar["tmc2_azimuth"], solar["tmc2_elevation"])]
                if rid == "region_001"
                else [("authoritative", azim, elev)]
            )
            try:
                for label, az, el in geoms:
                    shade = preview_hillshade(
                        crop_records[rid]["crop"].elevation_m,
                        az,
                        el,
                        working_gsd_m=60.0,
                        provenance_label=solar["elevation_provenance"],
                    )
                    out_png = previews_dir / f"{rid}_hillshade_{label}.npy"
                    # Diagnostic array stored as .npy (lossless, local-only); PNG
                    # rendering is left to the reviewer to avoid extra codecs.
                    np.save(str(out_png), np.asarray(shade, dtype=np.float32))
                    status["rendered"] = True
                    try:
                        status["preview_path"] = str(out_png.relative_to(repo_root))
                    except ValueError:
                        status["preview_path"] = str(out_png)
                    status["preview_shape"] = list(shade.shape)
                    status["preview_min"] = float(np.min(shade))
                    status["preview_max"] = float(np.max(shade))
                status["note"] = (
                    "region_001 diagnostic only; historical_repository_metadata, "
                    "not newly recovered PDS4 truth; no correspondence comparison."
                    if rid == "region_001" else "operator-supplied authoritative solar geometry."
                )
            except DEMIngestionError as exc:
                status["note"] = str(exc)
        elif rid not in crop_records:
            status["note"] = "No validated DEM crop available; hillshade not attempted."
        else:
            try:
                preview_hillshade(
                    crop_records[rid]["crop"].elevation_m,
                    solar.get("tmc2_azimuth"),
                    solar.get("tmc2_elevation"),
                    provenance_label=solar["elevation_provenance"],
                )
            except SolarElevationRequiredError as exc:
                status["note"] = str(exc)
        hillshade_status[rid] = status

    integrity = production_integrity(repo_root)

    # --- Decision (exactly one) ---
    any_tile_verified = any(
        e.get("checksum_status") == "VERIFIED" for e in tile_results.values()
    )
    all_tiles_present = all(
        e.get("found_path") is not None for e in tile_results.values()
    ) if tile_results else False
    all_crops_ok = (
        all_tiles_present
        and all(rid in crop_records for rid in ALL_REGION_IDS)
        and all(e.get("raster_error") is None for e in tile_results.values())
    )
    solar_complete = all(
        solar_status_for_region(rid, region_manifests, operator_solar)["renderable"]
        for rid in ALL_REGION_IDS
    )
    if hard_failure is not None:
        decision = "DEM_INGESTION_FAILED"
        decision_reason = hard_failure
    elif all_crops_ok and solar_complete and hillshade_status.get("region_001", {}).get("rendered"):
        decision = "DEM_INGESTION_VALIDATED"
        decision_reason = (
            "Both SLDEM2015 tiles verified, all six regions covered/cropped, "
            "authoritative solar geometry present, provenance recorded."
        )
    else:
        decision = "DEM_INGESTION_BLOCKED"
        missing_files = [e["tile_id"] for e in tile_results.values() if e.get("found_path") is None]
        missing_checksums = [
            e["tile_id"] for e in tile_results.values()
            if e.get("checksum_status") == "EXPECTED_CHECKSUM_NOT_SUPPLIED"
        ]
        parts = []
        if missing_files:
            parts.append(f"missing operator DEM files: {missing_files}")
        if missing_checksums:
            parts.append(f"checksums not supplied: {missing_checksums}")
        if all_tiles_present and not all_crops_ok:
            parts.append("crops incomplete (see per-region errors)")
        if not solar_complete:
            parts.append("authoritative solar elevation missing for regions 002-006")
        decision_reason = "; ".join(parts) or "operator inputs incomplete (see sections below)"

    write_report(
        report_path=report_path,
        repo_root=repo_root,
        dem_dir=dem_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        tile_results=tile_results,
        region_bounds=region_bounds,
        crop_records=crop_records,
        region_manifests=region_manifests,
        operator_solar=operator_solar,
        hillshade_status=hillshade_status,
        provenance_paths=provenance_paths,
        integrity=integrity,
        decision=decision,
        decision_reason=decision_reason,
    )
    print(f"Phase 14 validation complete: {decision}")
    print(f"Report: {report_path}")
    return 0


def write_report(**ctx) -> None:
    report_path: Path = ctx["report_path"]
    repo_root: Path = ctx["repo_root"]
    tile_results: dict = ctx["tile_results"]
    region_bounds: dict = ctx["region_bounds"]
    crop_records: dict = ctx["crop_records"]
    region_manifests: dict = ctx["region_manifests"]
    hillshade_status: dict = ctx["hillshade_status"]
    provenance_paths: dict = ctx["provenance_paths"]
    integrity: dict = ctx["integrity"]
    decision: str = ctx["decision"]
    decision_reason: str = ctx["decision_reason"]
    manifest: dict = ctx["manifest"]

    def rel(p: str | Path) -> str:
        try:
            return str(Path(p).relative_to(repo_root))
        except ValueError:
            return str(p)

    lines: list[str] = []
    lines.append("# Phase 14 — SLDEM2015 Verified DEM Ingestion & Read-Only Hillshade Preview")
    lines.append("")
    lines.append(f"**Date:** 2026-09-13  **Code:** `{CODE_VERSION}`  **Dataset:** `{DATASET_ID}`")
    lines.append(f"**Classification:** **{decision}**")
    lines.append(f"**Reason:** {decision_reason}")
    lines.append("")
    lines.append("> INGESTION VALIDATION ONLY. No correspondence matching, no model training,")
    lines.append("> no RF evaluation, no DEM/hillshade integration into the registration pipeline.")
    lines.append("> Production matcher, candidate-generation defaults, RF model, ground truth,")
    lines.append("> and held-out groups untouched (§K). `dem_512.png` never used; no DEM or solar")
    lines.append("> geometry fabricated.")
    lines.append("")
    lines.append("## A. Input files found/missing")
    lines.append("")
    lines.append(f"Resolved DEM directory: `{rel(ctx['dem_dir'])}` "
                 f"(CLI `--dem-dir` > `$CH2X_DEM_DIR` > `data/external/dem/`).")
    lines.append(f"Manifest: `{rel(ctx['manifest_path'])}`.")
    lines.append("")
    lines.append("| Tile | Expected file | Found |")
    lines.append("| --- | --- | --- |")
    for tile in manifest.get("tiles", []):
        tid = tile.get("tile_id")
        entry = tile_results.get(tid, {})
        found = entry.get("found_path")
        lines.append(f"| `{tid}` | `{tile.get('expected_filename')}` | "
                     f"{'`' + rel(found) + '`' if found else '**MISSING**'} |")
    if not tile_results:
        lines.append("| (no tiles contracted) | — | — |")
    lines.append("")
    lines.append("## B. SHA-256 verification")
    lines.append("")
    lines.append("| Tile | Status | SHA-256 (computed) |")
    lines.append("| --- | --- | --- |")
    for tid, entry in tile_results.items():
        lines.append(f"| `{tid}` | `{entry.get('checksum_status')}` | "
                     f"`{entry.get('checksum') or '—'}` |")
    lines.append("")
    lines.append("Operator checksum source: manifest `expected_sha256`, overridden by "
                 "`SLDEM_TILE_315_360_SHA256` / `SLDEM_TILE_225_270_SHA256` when exported. "
                 "`EXPECTED_CHECKSUM_NOT_SUPPLIED` blocks validation (modified tiles are "
                 "never silently accepted).")
    lines.append("")
    lines.append("## C. Raster metadata (verbatim, pre-crop; no reprojection performed)")
    lines.append("")
    for tid, entry in tile_results.items():
        lines.append(f"### {tid}")
        lines.append("")
        raster = entry.get("raster")
        if not raster:
            lines.append(f"Not available: `{entry.get('raster_error') or entry.get('checksum_status')}`.")
            lines.append("")
            continue
        for key in ["path", "driver", "width", "height", "count", "dtype", "crs_string",
                    "transform", "bounds_west", "bounds_south", "bounds_east", "bounds_north",
                    "res_x_deg", "res_y_deg", "pixels_per_degree", "nodata",
                    "unit_conversion_applied"]:
            lines.append(f"* `{key}`: `{raster.get(key)}`")
        lines.append("")
    lines.append("## D. Coverage verification (manifest footprints vs DEM bounds)")
    lines.append("")
    lines.append("| Region | Requested bounds (W,S,E,N) | DEM tile | DEM bounds | Contained | Margin px (W,E,S,N) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for tid, entry in tile_results.items():
        for rid, cov in entry.get("coverage", {}).items():
            req = cov["requested_bounds"]
            m = cov["margin_pixels"]
            lines.append(
                f"| `{rid}` | ({req['west_lon']:.6f}, {req['south_lat']:.6f}, "
                f"{req['east_lon']:.6f}, {req['north_lat']:.6f}) | `{cov['dem_tile']}` | "
                f"({cov['dem_bounds'][0]:.3f}, {cov['dem_bounds'][1]:.3f}, "
                f"{cov['dem_bounds'][2]:.3f}, {cov['dem_bounds'][3]:.3f}) | "
                f"{'YES' if cov['containment'] else '**NO**'} | "
                f"({fmt_m(m.get('west'))}, {fmt_m(m.get('east'))}, "
                f"{fmt_m(m.get('south'))}, {fmt_m(m.get('north'))}) |"
            )
    uncovered = [rid for rid in region_bounds if rid not in crop_records]
    if uncovered:
        lines.append("")
        lines.append(f"Regions without a validated crop (no coverage attempted or failed): "
                     f"`{uncovered}`.")
    lines.append("")
    lines.append("## E. Per-region crop statistics (diagnostics only)")
    lines.append("")
    lines.append("| Region | Shape | Min (m) | Max (m) | Mean (m) | Median (m) | Nodata frac | Finite frac |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for rid in ALL_REGION_IDS:
        rec = crop_records.get(rid)
        if not rec:
            err = ""
            for entry in tile_results.values():
                if rid in entry.get("crops", {}) and "error" in entry["crops"][rid]:
                    err = entry["crops"][rid]["error"]
            lines.append(f"| `{rid}` | — | — | — | — | — | — | — |"
                         + (f" `{err}`" if err else ""))
            continue
        s = rec["stats"]
        lines.append(
            f"| `{rid}` | {s['cropped_shape']} | {s['min_elevation_m']:.2f} | "
            f"{s['max_elevation_m']:.2f} | {s['mean_elevation_m']:.2f} | "
            f"{s['median_elevation_m']:.2f} | {s['nodata_fraction']:.4f} | "
            f"{s['finite_pixel_fraction']:.4f} |"
        )
    lines.append("")
    lines.append("Unusual terrain values are NOT interpreted as errors without evidence. "
                 "Crops with nodata fraction > 0.10 fail rather than being filled "
                 "(no interpolation in this phase).")
    lines.append("")
    lines.append("## F. Solar metadata status")
    lines.append("")
    lines.append("| Region | OHRC az | TMC-2 az | OHRC elev | TMC-2 elev | Elevation provenance | Renderable |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rid in ALL_REGION_IDS:
        st = hillshade_status[rid]["solar"]
        lines.append(
            f"| `{rid}` | {st.get('ohrc_azimuth')} | {st.get('tmc2_azimuth')} | "
            f"{st.get('ohrc_elevation')} | {st.get('tmc2_elevation')} | "
            f"`{st.get('elevation_provenance')}` | {'YES' if st.get('renderable') else 'NO'} |"
        )
    lines.append("")
    lines.append("## G. Region 001 hillshade preview status")
    lines.append("")
    st1 = hillshade_status.get("region_001", {})
    solar1 = st1.get("solar", {})
    lines.append(f"* Historical elevations (repository evidence only): OHRC "
                 f"{solar1.get('ohrc_elevation')}°, TMC-2 {solar1.get('tmc2_elevation')}° "
                 f"(`historical_repository_metadata` from `triplets.json[1]`; NOT PDS4 truth).")
    note1 = str(st1.get("note") or "").rstrip()
    if note1 and not note1.endswith("."):
        note1 += "."
    lines.append(f"* Rendered: `{st1.get('rendered')}`. "
                 f"Preview: `{st1.get('preview_path') or '—'}`. Note: {note1}")
    lines.append("* No comparison to correspondence performance was performed (out of scope).")
    lines.append("")
    lines.append("## H. Missing metadata for regions 002–006")
    lines.append("")
    lines.append("* Solar elevation for regions 002–006 is **unset** "
                 "(`unset_pending_authoritative_pds4_label`). Previews refused with "
                 f"`{SOLAR_ELEVATION_REQUIRED}` — nothing inferred from azimuth, brightness, or terrain.")
    lines.append("* Required Chandrayaan-2 PDS4 detached labels (operator-supplied via ISRO PRADAN):")
    lines.append("  * regions 001–004: `ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml` + "
                 "`ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml`")
    lines.append("  * regions 005–006: `ch2_ohr_ncp_20220914T0835371412_d_img_d32.xml` + "
                 "`ch2_tmc_ncf_20191125T0749024661_d_img_d18.xml`")
    lines.append("* Future interface: `PDS4Metadata` in `ML_model/dem_ingestion.py` "
                 "(source_image_id, pds4_label_path, observation_time_utc, solar_azimuth_deg, "
                 "solar_elevation_deg, spacecraft_position, target_body, footprint, provenance); "
                 "authoritative elevations may additionally be passed via `--solar-json` "
                 "without changing DEM validation logic.")
    lines.append("")
    lines.append("## I. Provenance records")
    lines.append("")
    if provenance_paths:
        for rid, p in provenance_paths.items():
            lines.append(f"* `{rid}`: `{p}`")
    else:
        lines.append("No provenance records issued (no successfully validated DEM crop). "
                     "Raw DEM binaries are never committed to git; only manifests, checksums, "
                     "and `evaluation/phase14_provenance/*.json` are tracked.")
    lines.append("")
    lines.append("## J. Validation decision")
    lines.append("")
    lines.append(f"**{decision}** — {decision_reason}")
    lines.append("")
    lines.append("A successful Phase 14 means only: real SLDEM2015 terrain can now be "
                 "reproducibly ingested and validated. It does NOT establish that DEM-based "
                 "hillshade improves cross-sensor correspondence (separate reviewed phase).")
    lines.append("")
    lines.append("## K. Production integrity appendix (scope guard)")
    lines.append("")
    for rel_path in PRODUCTION_GUARD_PATHS:
        info = integrity.get(rel_path, {})
        if info.get("modified") is None:
            lines.append(f"* `{rel_path}`: check unavailable (`{info.get('error')}`).")
        else:
            lines.append(f"* `{rel_path}`: {'**MODIFIED (SCOPE FAILURE)**' if info['modified'] else 'unmodified'}.")
    lines.append("")
    lines.append("Working-tree status (`git status --short`, may include this phase's own new files):")
    lines.append("```")
    lines.append(str(integrity.get("_git_status_short", ""))[:2000])
    lines.append("```")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_m(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


if __name__ == "__main__":
    sys.exit(main())
