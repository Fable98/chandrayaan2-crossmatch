# Phase 12 — Verified Real DEM & Solar-Geometry Ingestion Audit

**Date:** September 13, 2026
**Status:** Complete — Data Validation & Feasibility Audit Only
**Classification / Decision:** **BLOCKED**
**Production Integrity:** Frozen and Unmodified (`half_p=8`, `use_goa_preselection=False`, `ML_model/ai_verifier_model.pkl` unchanged)

---

## A. Executive Conclusion

Phase 12 audited the repository's data, metadata, and code infrastructure to evaluate the feasibility of ingesting verified real lunar altimetry (LOLA / SLDEM2015) for `region_001` through `region_006` and extracting exact solar elevation from PDS4 XML labels.

### Key Audit Findings

1. **Geographic Footprints Are Well-Defined**:
   - The repository contains exact selenographic bounding box coordinates (`west_lon`, `east_lon`, `south_lat`, `north_lat`) for all six regions (`region_001` through `region_006`) in `data_preprocessing_pipeline/processed_triplets/region_*/manifest.json`.
   - The spatial footprints are consistently referenced to the lunar Equirectangular projection (`+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400`).

2. **Authoritative Altimetry Product IDs Are Absent**:
   - Zero product identifiers, PDS dataset IDs, or download URLs for authoritative LOLA (e.g. `LOLA_GDR`, `LDEM`) or SLDEM2015 products exist anywhere in the repository.
   - The expansion registry (`data_preprocessing_pipeline/lro_cdr_registry.json`) is strictly dedicated to LRO NAC optical panchromatic frames (`M1417670274LC`, `M1413636095LC`) and contains no altimetry slots.

3. **Solar Elevation Metadata Is Incomplete**:
   - Solar azimuth (`sun_azimuth_deg`) is documented across all regions.
   - Solar elevation (`sun_elevation_deg`) is `None` in the manifests for all six regions.
   - For `region_001`, historical elevation values ($14.06^\circ$ OHRC, $48.75^\circ$ TMC-2) exist in `data_preprocessing_pipeline/triplets.json`.
   - However, the raw PDS4 XML labels for `region_002` through `region_006` were external files (`<external-downloads-dir>`) processed offline and never committed to git. Consequently, exact solar elevation cannot be extracted from within the repository for regions 002–006.

4. **Existing Code Cannot Ingest Altimetry Without Interface Changes**:
   - `LRODataHandler` in `backend/data/lro_validator.py` is a point-elevation sampler; it cannot crop, warp, or resample a 2D DEM grid to match optical image dimensions, nor does it have download capability.
   - `lro_ode_client.py` is hardcoded for optical camera search (`iid="lroc"`, `pt="EDRNAC"`) and post-processes via PDS3 image decoders designed for line-reversed optical swaths, not binary elevation rasters.

**Final Decision**: **BLOCKED**. Critical product identifiers and raw PDS4 labels are missing from the repository. Ingestion cannot proceed until authoritative product IDs and elevation metadata are provided.

---

## B. Existing Ingestion Infrastructure

We audited the six key components of the repository's data-handling infrastructure:

| Component | Location | Current Capability | Limitations for DEM Ingestion |
| :--- | :--- | :--- | :--- |
| **`LRODataHandler`** | [`backend/data/lro_validator.py`](backend/data/lro_validator.py) | Loads local `.tif` or `.img` via `rasterio`, inspects metadata, and queries elevation at `(lat, lon)` coordinates. | Point-sampler only. Lacks 2D spatial window cropping, GSD resampling, and network download capability. Assumes input coordinates directly match raster transform. |
| **`lro_ode_client.py`** | [`ML_model/lro_ode_client.py`](ML_model/lro_ode_client.py) | Queries Washington University ODE REST API for products overlapping bounding boxes; downloads and caches files. | Hardcoded for LRO NAC optical cameras (`ihid="lro"`, `iid="lroc"`, `pt="EDRNAC"`). Post-download handler expects optical PDS3 images (`decode_pds3_img_to_array`), incompatible with LOLA GDR or SLDEM GeoTIFFs. |
| **`manifest.json`** | `data_preprocessing_pipeline/processed_triplets/region_*/` | Records product IDs, native GSDs, solar azimuth, bounding boxes, and image paths. | `ohrc_sun_elevation_deg` and `tmc2_sun_elevation_deg` are missing / set to `None`. No DEM source product IDs. |
| **PDS4 Parser** | [`data/ingestion/pds4_reader.py`](data/ingestion/pds4_reader.py) | Parses XML DOM for axis dimensions, GSD, `geom:sun_azimuth`, `geom:sun_elevation`, and SPICE kernels. | Fully functional parser, but the PDS4 XML files for development regions 002–006 are not present in the repository. |
| **Georeferencing Logic** | `data_preprocessing_pipeline/scripts/make_demo_regions.py` | Projects between lunar geographic CRS and Moon Equirectangular (`+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400`). | Robust for optical tiles, but lacks reprojection pipeline for global LOLA grids ($60\text{–}118\text{ m/px}$) to local high-res grids. |
| **DEM Hillshade Engine** | [`ML_model/matcher_cfog.py`](ML_model/matcher_cfog.py#L1070-L1255) | `compute_dem_cast_shadows` (ray-marched shadows) and `render_synthetic_shaded_relief` (Horn slope/aspect + Lambertian illumination). | Fully implemented and mathematically verified; currently idle awaiting calibrated DEM inputs. |

---

## C. Region Footprint Availability

The repository contains exact geographic bounding box metadata for all six development regions in their respective `manifest.json` files:

| Region | Longitude Bounds (deg E) | Latitude Bounds (deg N) | Dimensions ($512 \times 512$ tile) | Source Product ID (OHRC) | Target Product ID (TMC-2) |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`region_001`** | $[336.484646^\circ, 336.589455^\circ]$ | $[-3.374861^\circ, -3.248733^\circ]$ | $\approx 3.18 \times 3.82\text{ km}$ | `ch2_ohr_ncp_20210405t1606536730_d_img_d18` | `ch2_tmc_ncf_20250807t1904346039_d_img_d18` |
| **`region_002`** | $[336.484646^\circ, 336.589455^\circ]$ | $[-3.206690^\circ, -3.080562^\circ]$ | $\approx 3.18 \times 3.82\text{ km}$ | `ch2_ohr_ncp_20210405t1606536730_d_img_d18` | `ch2_tmc_ncf_20250807t1904346039_d_img_d18` |
| **`region_003`** | $[336.484646^\circ, 336.589455^\circ]$ | $[-3.038519^\circ, -2.912390^\circ]$ | $\approx 3.18 \times 3.82\text{ km}$ | `ch2_ohr_ncp_20210405t1606536730_d_img_d18` | `ch2_tmc_ncf_20250807t1904346039_d_img_d18` |
| **`region_004`** | $[336.484646^\circ, 336.589455^\circ]$ | $[-2.870348^\circ, -2.744219^\circ]$ | $\approx 3.18 \times 3.82\text{ km}$ | `ch2_ohr_ncp_20210405t1606536730_d_img_d18` | `ch2_tmc_ncf_20250807t1904346039_d_img_d18` |
| **`region_005`** | $[234.396774^\circ, 234.528638^\circ]$ | $[+4.863173^\circ, +5.067201^\circ]$ | $\approx 4.00 \times 6.18\text{ km}$ | `ch2_ohr_ncp_20220914t0835371412_d_img_d32` | `ch2_tmc_ncf_20191125t0749024661_d_img_d18` |
| **`region_006`** | $[234.396774^\circ, 234.528638^\circ]$ | $[+5.148812^\circ, +5.352839^\circ]$ | $\approx 4.00 \times 6.18\text{ km}$ | `ch2_ohr_ncp_20220914t0835371412_d_img_d32` | `ch2_tmc_ncf_20191125t0749024661_d_img_d18` |

**Conclusion**: Geographic footprints are complete and ready for spatial indexing.

---

## D. Required LOLA / SLDEM2015 Products

A search of the entire repository confirmed:
- Zero LOLA product IDs (e.g., `LOLA_GDR`, `LDEM_512`, `SLDEM2015_...`) exist in any config, script, or documentation.
- The repository does not store pre-determined product IDs or direct download URLs for elevation data.
- **Missing Inputs**: Authoritative NASA/PDS or USGS product identifiers for the LOLA / SLDEM2015 tiles covering:
  - Lon $336.48^\circ\text{–}336.59^\circ\text{ E}$, Lat $-3.38^\circ\text{–}-2.74^\circ\text{ N}$ (Regions 001–004, Sinus Medii / equatorial near-side).
  - Lon $234.39^\circ\text{–}234.53^\circ\text{ E}$, Lat $+4.86^\circ\text{–}+5.35^\circ\text{ N}$ (Regions 005–006, equatorial highland).

In accordance with Hard Invariant 4 (*"Do NOT invent product IDs or URLs"*), we do not synthesize or assume external filenames.

---

## E. DEM Format & Georeferencing Requirements

To integrate seamlessly with the existing `LRODataHandler` and `render_synthetic_shaded_relief`, any candidate DEM raster must satisfy the following technical specification:

1. **Raster Format**: GeoTIFF (`.tif`) or PDS labeled raster (`.img` + `.lbl`).
2. **Band Structure**: Single-band 2D array, `float32` or `int16`/`int32` with scale/offset.
3. **Values**: Calibrated physical elevation in meters relative to lunar reference sphere ($R = 1,737,400.0\text{ m}$).
4. **Coordinate Reference System (CRS)**:
   - Moon Equirectangular (`+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m`) or Lunar Geographic (`+proj=longlat +a=1737400 +b=1737400`).
   - Valid GDAL geotransform mapping pixel indices $(c, r)$ to CRS coordinates $(x, y)$.
5. **Spatial Resolution**: Native LOLA GDR ($118.5\text{ m/px}$, 256 pixels/deg) or SLDEM2015 ($60\text{ m/px}$, 512 pixels/deg).
6. **NoData Value**: Finite sentinel (e.g. `-32768.0` or `np.nan`), explicitly specified in metadata.

---

## F. Solar-Geometry Metadata Availability

| Information | Status in Repo | Source File / Details |
| :--- | :---: | :--- |
| **Solar Azimuth (`region_001`–`006`)** | **Available** | Stored in `processed_triplets/region_*/manifest.json` (`ohrc_sun_azimuth_deg`, `tmc2_sun_azimuth_deg`). |
| **Solar Elevation (`region_001`)** | **Partially Available** | Documented in `data_preprocessing_pipeline/triplets.json`: $14.063051^\circ$ (OHRC), $48.753985^\circ$ (TMC-2). |
| **Solar Elevation (`region_002`–`006`)** | **Unavailable** | Set to `None` in manifests; not recorded in `triplets.json`. |
| **PDS4 XML Labels (`region_001`–`006`)** | **Unavailable in Git** | Raw XML files were located in external download paths (`<external-downloads-dir>`) during offline tiling and were not committed. |
| **SPICE Kernels** | **Unavailable** | No SPICE kernels (`.bsp`, `.bc`, `.ti`) exist locally. |

---

## G. Missing Data / Metadata Summary

The following required inputs are currently missing from the repository:

1. **Authoritative Altimetry Products**: Official PDS product IDs or direct download URLs for LOLA/SLDEM2015 rasters covering regions 001–004 and regions 005–006.
2. **PDS4 XML Labels**: Calibrated PDS4 observational labels for the source OHRC and target TMC-2 images for regions 002–006.
3. **Solar Elevation Angles**: Verified sun elevation angles for regions 002–006.
4. **Altimetry Ingestion Logic**: Extension of `lro_ode_client.py` to support ODE instrument ID `"lola"` and automated GeoTIFF spatial window clipping.

---

## H. Minimum Implementation Needed (When Unblocked)

When authoritative product IDs and elevation angles are supplied, the minimum required code changes will be:

1. **`lro_ode_client.py` Extension**:
   - Add parameter `instrument_id: str = "lroc"` with opt-in support for `"lola"`.
   - When querying LOLA, set `"iid": "lola"`, `"pt": "GDR"`.
   - Add a lightweight GeoTIFF window cropper using `rasterio.windows.from_bounds` to clip the regional bounding box from the global LOLA raster.
2. **Manifest Schema Update**:
   - Populate `ohrc_sun_elevation_deg`, `tmc2_sun_elevation_deg`, and `dem_product_id` in `manifest.json`.
3. **Elevation Unit Normalization**:
   - Ensure cropped elevation is passed to `render_synthetic_shaded_relief` as physical elevation in meters (`float32`), not uncalibrated `uint8`.

---

## I. Data Provenance & Reproducibility Requirements

To prevent recurrence of the 2026-09-11 failure mode (where synthetic DEM proxies corrupted benchmark interpretability), any future ingestion must enforce:

1. **Strict Provenance Labeling**: Ingested rasters must be tagged with `dem_provenance="authoritative_pds_lola"` or `"sldem2015"`.
2. **Integrity Checksums**: SHA-256 hashes of downloaded rasters must be recorded in an immutable registry (analogous to `lro_cdr_registry.json`).
3. **Zero In-Memory Fabrication**: Never synthesize elevation from optical gradients (`cv2.Sobel`) or claim synthetic stereo disparity as a real DEM.

---

## J. Decision

# **DECISION: BLOCKED**

**Rationale**:
- Sufficient geographic footprint metadata exists (Lon/Lat bounds are complete).
- The rendering infrastructure (`render_synthetic_shaded_relief`) is fully functional.
- **HOWEVER**, ingestion cannot execute because:
  1. Authoritative LOLA/SLDEM2015 product IDs or URLs are not in the repository.
  2. Raw PDS4 XML labels for regions 002–006 are absent, leaving solar elevation unavailable.
  3. `lro_ode_client.py` currently supports only optical LROC NAC frames.

---

## Test Results & Production Integrity Confirmation

### 1. Test Suite Verification
- `python3 -m pytest -q`: **258 passed, 2 skipped** (100% pass rate).
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** (100% pass rate).

### 2. Model & Hyperparameter Freeze Confirmation
- `git diff -- ML_model/ai_verifier_model.pkl`: **Empty** (0 bytes modified).
- Production candidate matching defaults remain strictly frozen: `half_p=8`, `use_goa_preselection=False`.
- `ground_truth_matches.json`: Untouched.
- Held-out test groups: Untouched.

### 3. Files Created / Modified
- Created: `evaluation/phase12_dem_ingestion_audit.md` (this report).
- Modified: None. Production code is strictly unchanged.
