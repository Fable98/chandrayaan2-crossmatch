# Phase 14 — SLDEM2015 Verified DEM Ingestion & Read-Only Hillshade Preview

**Date:** 2026-09-13  **Code:** `phase14-v1`  **Dataset:** `LRO-L-LOLA-4-GDR-V1.0`
**Classification:** **DEM_INGESTION_BLOCKED**
**Reason:** missing operator DEM files: ['SLDEM2015_512_30S_00S_315_360', 'SLDEM2015_512_00N_30N_225_270']; authoritative solar elevation missing for regions 002-006

> INGESTION VALIDATION ONLY. No correspondence matching, no model training,
> no RF evaluation, no DEM/hillshade integration into the registration pipeline.
> Production matcher, candidate-generation defaults, RF model, ground truth,
> and held-out groups untouched (§K). `dem_512.png` never used; no DEM or solar
> geometry fabricated.

## A. Input files found/missing

Resolved DEM directory: `data/external/dem` (CLI `--dem-dir` > `$CH2X_DEM_DIR` > `data/external/dem/`).
Manifest: `data/external/dem/dem_manifest.json`.

| Tile | Expected file | Found |
| --- | --- | --- |
| `SLDEM2015_512_30S_00S_315_360` | `SLDEM2015_512_30S_00S_315_360.jp2` | **MISSING** |
| `SLDEM2015_512_00N_30N_225_270` | `SLDEM2015_512_00N_30N_225_270.jp2` | **MISSING** |

## B. SHA-256 verification

| Tile | Status | SHA-256 (computed) |
| --- | --- | --- |
| `SLDEM2015_512_30S_00S_315_360` | `FILE_MISSING` | `—` |
| `SLDEM2015_512_00N_30N_225_270` | `FILE_MISSING` | `—` |

Operator checksum source: manifest `expected_sha256`, overridden by `SLDEM_TILE_315_360_SHA256` / `SLDEM_TILE_225_270_SHA256` when exported. `EXPECTED_CHECKSUM_NOT_SUPPLIED` blocks validation (modified tiles are never silently accepted).

## C. Raster metadata (verbatim, pre-crop; no reprojection performed)

### SLDEM2015_512_30S_00S_315_360

Not available: `FILE_MISSING`.

### SLDEM2015_512_00N_30N_225_270

Not available: `FILE_MISSING`.

## D. Coverage verification (manifest footprints vs DEM bounds)

| Region | Requested bounds (W,S,E,N) | DEM tile | DEM bounds | Contained | Margin px (W,E,S,N) |
| --- | --- | --- | --- | --- | --- |

Regions without a validated crop (no coverage attempted or failed): `['region_001', 'region_002', 'region_003', 'region_004', 'region_005', 'region_006']`.

## E. Per-region crop statistics (diagnostics only)

| Region | Shape | Min (m) | Max (m) | Mean (m) | Median (m) | Nodata frac | Finite frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `region_001` | — | — | — | — | — | — | — |
| `region_002` | — | — | — | — | — | — | — |
| `region_003` | — | — | — | — | — | — | — |
| `region_004` | — | — | — | — | — | — | — |
| `region_005` | — | — | — | — | — | — | — |
| `region_006` | — | — | — | — | — | — | — |

Unusual terrain values are NOT interpreted as errors without evidence. Crops with nodata fraction > 0.10 fail rather than being filled (no interpolation in this phase).

## F. Solar metadata status

| Region | OHRC az | TMC-2 az | OHRC elev | TMC-2 elev | Elevation provenance | Renderable |
| --- | --- | --- | --- | --- | --- | --- |
| `region_001` | 269.646098 | 108.866212 | 14.063051 | 48.753985 | `historical_repository_metadata` | YES |
| `region_002` | 269.646098 | 108.866212 | None | None | `unset_pending_authoritative_pds4_label` | NO |
| `region_003` | 269.646098 | 108.866212 | None | None | `unset_pending_authoritative_pds4_label` | NO |
| `region_004` | 269.646098 | 108.866212 | None | None | `unset_pending_authoritative_pds4_label` | NO |
| `region_005` | 89.392682 | 251.647896 | None | None | `unset_pending_authoritative_pds4_label` | NO |
| `region_006` | 89.392682 | 251.647896 | None | None | `unset_pending_authoritative_pds4_label` | NO |

## G. Region 001 hillshade preview status

* Historical elevations (repository evidence only): OHRC 14.063051°, TMC-2 48.753985° (`historical_repository_metadata` from `triplets.json[1]`; NOT PDS4 truth).
* Rendered: `False`. Preview: `—`. Note: No validated DEM crop available; hillshade not attempted.
* No comparison to correspondence performance was performed (out of scope).

## H. Missing metadata for regions 002–006

* Solar elevation for regions 002–006 is **unset** (`unset_pending_authoritative_pds4_label`). Previews refused with `SOLAR_ELEVATION_REQUIRED` — nothing inferred from azimuth, brightness, or terrain.
* Required Chandrayaan-2 PDS4 detached labels (operator-supplied via ISRO PRADAN):
  * regions 001–004: `ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml` + `ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml`
  * regions 005–006: `ch2_ohr_ncp_20220914T0835371412_d_img_d32.xml` + `ch2_tmc_ncf_20191125T0749024661_d_img_d18.xml`
* Future interface: `PDS4Metadata` in `ML_model/dem_ingestion.py` (source_image_id, pds4_label_path, observation_time_utc, solar_azimuth_deg, solar_elevation_deg, spacecraft_position, target_body, footprint, provenance); authoritative elevations may additionally be passed via `--solar-json` without changing DEM validation logic.

## I. Provenance records

No provenance records issued (no successfully validated DEM crop). Raw DEM binaries are never committed to git; only manifests, checksums, and `evaluation/phase14_provenance/*.json` are tracked.

## J. Validation decision

**DEM_INGESTION_BLOCKED** — missing operator DEM files: ['SLDEM2015_512_30S_00S_315_360', 'SLDEM2015_512_00N_30N_225_270']; authoritative solar elevation missing for regions 002-006

A successful Phase 14 means only: real SLDEM2015 terrain can now be reproducibly ingested and validated. It does NOT establish that DEM-based hillshade improves cross-sensor correspondence (separate reviewed phase).

## K. Production integrity appendix (scope guard)

* `ML_model/matcher_cfog.py`: unmodified.
* `ML_model/ai_verifier.py`: unmodified.
* `ML_model/train_ai_verifier.py`: unmodified.
* `ML_model/ground_truth_matches.json`: unmodified.
* `ML_model/ai_verifier_model.pkl`: unmodified.

Working-tree status (`git status --short`, may include this phase's own new files):
```
M .gitignore
?? ML_model/dem_ingestion.py
?? data/external/
?? evaluation/phase14_dem_ingestion_validation.md
?? evaluation/phase14_provenance/
?? scripts/validate_phase14_dem.py
?? tests/test_phase14_dem_ingestion.py
```
