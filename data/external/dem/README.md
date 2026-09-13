# Phase 14 — Operator DEM Supply Contract (SLDEM2015 512 ppd)

This directory is the **only** accepted location for the two authoritative DEM tiles.
Nothing here is downloaded automatically. Nothing is fabricated when files are absent.

## Required files

| Tile (product ID) | Expected filename | Covers |
| --- | --- | --- |
| `SLDEM2015_512_30S_00S_315_360` | `SLDEM2015_512_30S_00S_315_360.jp2` | region_001–region_004 |
| `SLDEM2015_512_00N_30N_225_270` | `SLDEM2015_512_00N_30N_225_270.jp2` | region_005–region_006 |

Accepted alternates per tile: same stem with `.tif` or `.img` (see `dem_manifest.json`).
Place sidecars alongside when available: `*_AUX.XML`, `*_JP2.LBL`, `.LBL`, `.XML`.

**Explicitly rejected:** `dem_512.png` and any `*.png` (synthesized optical-gradient
product, NOT authoritative elevation). LOLA tiles are NOT accepted unless a later
reviewed phase explicitly authorises the control run.

## How to supply

1. Download the two tiles from the authoritative index recorded in
   `dem_manifest.json` (`https://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/`).
2. Copy them into this directory (or set `CH2X_DEM_DIR` to a custom directory
   containing them — the validator resolves `--dem-dir > $CH2X_DEM_DIR > data/external/dem/`).
3. Record provenance in `dem_manifest.json` (immutable-style: fill, do not rename keys):
   `expected_sha256`, `acquisition_download_date`, plus the source URL / version
   from the tile `.LBL`/`.XML`. Alternatively export per-tile checksum overrides:
   `SLDEM_TILE_315_360_SHA256`, `SLDEM_TILE_225_270_SHA256`.
4. Run: `python3 scripts/validate_phase14_dem.py [--dem-dir <dir>] [--manifest <json>]`

## Absent-file behaviour

If a file is absent the validator **fails clearly** (`DEM_INGESTION_BLOCKED`),
names the exact missing file, and stops before any hillshade rendering.
No placeholder data is ever generated.

## Git

Raw DEM binaries in this directory are ignored by git (see `.gitignore`).
Only `dem_manifest.json`, this README, checksums, and derived provenance records
under `evaluation/phase14_provenance/` are tracked.
