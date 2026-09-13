# Phase 11 — DEM/Hillshade Feasibility & Correspondence Diagnostic Report

**Date:** September 13, 2026
**Status:** Complete — Feasibility Audit & Diagnostic Only
**Classification:** **Classification C — DEM/Hillshade Not Currently Feasible**
**Production Integrity:** Frozen and Unmodified (`half_p=8`, `use_goa_preselection=False`, `ML_model/ai_verifier_model.pkl` unchanged)

---

## 1. Executive Summary

Phase 11 evaluated the feasibility of the Phase 10 recommended direction:
> *DEM Relief-Compensated Hillshade Synthesis / Global Solar Alignment: Instead of attempting to make OHRC and TMC-2 optical appearance directly invariant to an extreme ~160° solar illumination reversal, utilize terrain elevation models (DEM) and known ephemeris metadata (`sun_azimuth_deg`, `sun_elevation_deg`) to synthesize a reference hillshade under the target illumination conditions.*

Before building or integrating a DEM hillshade matcher into production, this phase performed a rigorous repository data audit to determine whether real DEMs and paired solar geometry exist for the development scenes.

### Key Audit Findings

1. **Absence of Real Digital Elevation Models (DEMs)**:
   - A complete search for elevation rasters (`.tif`, `.tiff`, `.img`, `.hgt`, `.nc`, `.npy`, `.npz`) revealed zero calibrated DEM files in the repository.
   - The only elevation-related files present are `dem_512.png` located in `data_preprocessing_pipeline/processed_triplets/region_001/` through `region_006/` and `sample_data/dem_sample.png`.
   - Inspection of `data_preprocessing_pipeline/scripts/make_demo_regions.py` (lines 163–171) revealed that these `dem_512.png` files are **not real elevation measurements**. They were fabricated using a 2D Sobel gradient proxy on the optical TMC-2 image (`grad_x = cv2.Sobel(tmc_raw)`), which was used to synthesize along-track stereo views and run block-matching disparity.
   - In accordance with the hard constraint (*"Do NOT substitute synthetic DEM data. Do NOT implement a fake hillshade pipeline"*), synthetic DEM proxies cannot be used for scientific validation.

2. **Incomplete Solar Geometry Metadata**:
   - Solar azimuth (`ohrc_sun_azimuth_deg`, `tmc2_sun_azimuth_deg`) is documented across regions in `manifest.json`.
   - Solar elevation (`sun_elevation_deg`) is `None` in the per-region manifests (`region_001` through `region_006`). While a historical entry in `data_preprocessing_pipeline/triplets.json` records $14.06^\circ$ (OHRC) and $48.75^\circ$ (TMC-2) for `region_001`, no elevation angles exist for other development regions.
   - For the 37 synthetic development cross-sensor pairs, the transformation was generated via `apply_antisolar_cross_sensor` with a 2D filter ($\Delta \theta \approx 160^\circ \pm 8^\circ$) lacking 3D solar incidence vectors or DEM elevation grids.

3. **Existing Infrastructure Is Ready but Awaiting Real Data**:
   - The repository already contains a high-quality, vectorized GIS hillshade engine in `ML_model/matcher_cfog.py`:
     - `compute_dem_cast_shadows`: Ray-marched cast shadow computation on 2D elevation arrays.
     - `render_synthetic_shaded_relief`: Horn (1981) 3×3 slope/aspect calculation + Lambertian shading + shadow modulation.
     - An opt-in hook (`allow_synthetic_reference=True`) was previously explored on 2026-09-11 and disabled because using fabricated DEM proxies corrupted benchmark interpretability.

4. **Official Classification**:
   - **Classification C — DEM/Hillshade Not Currently Feasible**.
   - As mandated by the phase protocol, execution stopped after the audit. No external data was automatically downloaded, no synthetic DEM was substituted, and production remains unchanged.

---

## 2. Repository Data Audit

We performed an exhaustive audit across the repository for DEM rasters, elevation archives, and observation metadata.

### 2.1 File Type Search
Command:
```bash
find . -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.img" -o -name "*.hgt" -o -name "*.nc" -o -name "*.npy" -o -name "*.npz" \)
```
**Results**:
- Output `.tif` files found: Exclusively `registered_source.tif` files in benchmark/dynamic test output directories (e.g. `benchmarks/registration_benchmark_output/*/registered_source.tif`). These are 2D optical images saved as GeoTIFFs after registration.
- Elevation rasters found: **0 files**.

### 2.2 Elevation & DEM Directory Search
Command:
```bash
find . -iname "*dem*" -o -iname "*elev*" -o -iname "*lola*" -o -iname "*topo*"
```
**Results**:
- `data_preprocessing_pipeline/processed_triplets/region_001/dem_512.png` through `region_006/dem_512.png`
- `sample_data/dem_sample.png`
- `backend/data/lro_validator.py` (`LRODataHandler` class)
- `ML_model/tmc_stereo.py` (`derive_dem_from_tmc_stereo`)
- `ML_model/matcher_cfog.py` (`compute_dem_cast_shadows`, `render_synthetic_shaded_relief`)

### 2.3 LOLA / LRO-Derived Elevation
- `backend/data/lro_validator.py` defines `LRODataHandler.load_dem()`, an interface designed to load LOLA GeoTIFFs via `rasterio` and query elevation in meters relative to the Moon's radius ($1,737,400\text{ m}$).
- However, **no LOLA DEM files (`.tif` / `.img`) are present** in `data/`, `data_preprocessing_pipeline/`, or `sample_data/`.

---

## 3. DEM Availability & Provenance Analysis

The repository contains `dem_512.png` in each of the 6 processed triplet folders. We inspected `data_preprocessing_pipeline/scripts/make_demo_regions.py` (lines 163–171) to determine how these files were created:

```python
# Extract from data_preprocessing_pipeline/scripts/make_demo_regions.py:
grad_x = cv2.Sobel(tmc_raw, cv2.CV_32F, 1, 0, ksize=3)
grad_norm = grad_x / (np.max(np.abs(grad_x)) + 1e-6)
relief_init = -cv2.GaussianBlur(grad_norm, (9, 9), 2.0) * 150.0
tmc_u8_temp = to_u8(tmc_raw)
fore_syn, aft_syn = generate_synthetic_stereo_views(tmc_u8_temp, relief_init, gsd_m=5.0)
stereo_res = derive_dem_from_tmc_stereo(fore_syn, aft_syn, img_nadir=tmc_u8_temp, gsd_m=5.0)
dem_u8 = stereo_res["dem_u8"]
```

### Critical Provenance Finding:
1. The stored `dem_512.png` is not a measured topographic product.
2. It is a **circular derivative of the optical TMC-2 2D image**, created by filtering the optical image with Sobel horizontal gradients, synthesizing fake stereo views, and estimating block disparity.
3. The values are stored as uncalibrated `uint8` values ($0\text{–}255$), with no physical elevation scale in meters, no georeferenced vertical datum, and no validation against ground truth altimetry.
4. Using this image to render a hillshade would simply produce a blurred, re-shaded version of the 2D TMC-2 optical gradients, violating the experimental integrity requirements.

---

## 4. Solar Geometry Availability

We audited all manifest and metadata files for solar angles:

| Region | Source Sensor | Target Sensor | Sun Azimuth Source | Sun Azimuth Target | Sun Elevation Source | Sun Elevation Target | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `region_001` | OHRC | TMC-2 | $269.65^\circ$ | $108.87^\circ$ | `None` (manifest) / $14.06^\circ$ (triplets.json) | `None` (manifest) / $48.75^\circ$ (triplets.json) | $\Delta\text{az} = 160.78^\circ$ |
| `region_002` | OHRC | TMC-2 | $269.65^\circ$ | $108.87^\circ$ | `None` | `None` | Slice of 001 bundle |
| `region_003` | OHRC | TMC-2 | $269.65^\circ$ | $108.87^\circ$ | `None` | `None` | Slice of 001 bundle |
| `region_004` | OHRC | TMC-2 | $269.65^\circ$ | $108.87^\circ$ | `None` | `None` | Slice of 001 bundle |
| `region_005` | OHRC | TMC-2 | $89.39^\circ$ | $251.65^\circ$ | `None` | `None` | $\Delta\text{az} = 162.26^\circ$ |
| `region_006` | OHRC | TMC-2 | $89.39^\circ$ | $251.65^\circ$ | `None` | `None` | Slice of 005 bundle |
| 37 Dev Groups | OHRC | Synthetic Cross | Undefined | $\Delta\text{az} \approx 160^\circ \pm 8^\circ$ | Undefined | Undefined | 2D image filter only |

**Conclusion**: Solar azimuth is well-cataloged, but solar elevation is missing from all region manifests, and physical ephemeris geometry does not exist for the synthetic development cross-sensor dataset.

---

## 5. Existing Terrain / Hillshade Infrastructure

The codebase already possesses a complete, standalone GIS hillshade and ray-marching shadow engine in [`ML_model/matcher_cfog.py`](ML_model/matcher_cfog.py#L1070-L1255):

1. **Ray-Marched Cast Shadows** ([`compute_dem_cast_shadows`](ML_model/matcher_cfog.py#L1099-L1165)):
   - Traces solar sightlines along $\text{azimuth}$ and $\text{elevation}$ through 2D elevation grids.
   - Computes geometric line-of-sight occlusion: $z_{\text{terrain}} > z_{\text{base}} + d \cdot \tan(\text{elevation})$.
   - Returns float32 shadow mask $\in [0, 1]$ with edge smoothing for anti-aliasing.
2. **GIS Shaded Relief** ([`render_synthetic_shaded_relief`](ML_model/matcher_cfog.py#L1167-L1254)):
   - Computes Horn (1981) $3 \times 3$ elevation gradients normalized by physical pixel GSD ($dz/dx, dz/dy$).
   - Calculates local surface slope and aspect: $\text{slope} = \arctan(\sqrt{dzdx^2 + dzdy^2})$, $\text{aspect} = \arctan2(dzdx, -dzdy)$.
   - Evaluates Lambertian illumination: $I = \cos(\text{zenith})\cos(\text{slope}) + \sin(\text{zenith})\sin(\text{slope})\cos(\text{azimuth} - \text{aspect})$.
   - Modulates with ray-marched cast shadows.
3. **Previous Production Audit Note** ([`matcher_cfog.py:3435`](ML_model/matcher_cfog.py#L3435)):
   ```python
   # Rationale, measured 2026-09-11: silent swapping made benchmark outcomes uninterpretable
   # (002/004 success->fail, triplet_new_2022 fail->success) by matching
   # OHRC against a FABRICATED reference while metrics read as if real.
   # Synthetic-data matching is a legitimate experiment, never a default.
   ```
This note confirms that earlier in development, attempting to use the uncalibrated `dem_512.png` proxy damaged benchmark performance and masked genuine failures.

---

## 6. Audit of Scale, CRS, and Coordinate Systems

- **Optical Tiles**: $512 \times 512$ pixel PNGs covering approximately $3.18 \times 3.82\text{ km}$ at $\approx 6\text{–}8\text{ m/px}$ common display scale.
- **Georeferencing**: Manifest bounds specify lunar Equirectangular projection (`+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs`).
- **Real Lunar DEM Standards**: External LOLA/SLDEM2015 data is distributed in polar stereographic or equirectangular rasters at $60\text{ m/px}$ (SLDEM2015) or $118\text{ m/px}$ (LOLA PDS).
- **Resolution Gap**: A real lunar DEM ($60\text{ m/px}$) across a $3\text{ km}$ tile contains only $\approx 50 \times 50$ elevation grid points. Resampling a $50 \times 50$ elevation grid to $512 \times 512$ optical resolution would produce smooth macroscopic slopes, but would lack the crater rim details required for local feature correspondence.

---

## 7. Sections 5–9: Execution Status Under Protocol

Per Section 2 of the Phase 11 instruction:
> *"Classify the repository as: ... C — Required data is absent ... If classification C, STOP after the audit and report exactly what is missing. Do NOT download external data automatically. Do NOT substitute synthetic DEM data. Do NOT implement a fake hillshade pipeline."*

Because real DEM rasters are absent and synthetic proxies are strictly forbidden:
- **Section 5 (Selected Development Case)**: Not selected; no real DEM exists to pair with any development optical image.
- **Section 6 (Hillshade Synthesis)**: Not executed with synthetic proxies.
- **Section 7 (Correspondence Diagnostic)**: Not executed.
- **Section 8 (Geometric Scale & Resampling)**: Audited and documented in Section 6 above.
- **Section 9 (Null Model Comparison)**: Not executed.

Proceeding with synthetic `dem_512.png` would constitute a "fake hillshade pipeline" operating on fabricated references, which was explicitly forbidden.

---

## 8. Outcome Classification (Section 12)

### **Official Classification: C — DEM/Hillshade Not Currently Feasible**

**Scientific Rationale**:
1. **No Real Elevation Data**: Zero real DEM rasters exist in the repository. The available `dem_512.png` files are synthetic Sobel derivatives of optical images.
2. **Incomplete Solar Ephemeris**: Solar elevation angles are absent (`None`) in region manifests, and physical 3D solar incidence is undefined for the 37 synthetic development cross-sensor pairs.
3. **Hard Constraint Compliance**: Automatic downloading of external DEMs and substitution of synthetic elevation arrays were strictly prohibited.
4. **Previous Empirical Finding**: Using synthetic DEM proxies was previously proven (2026-09-11) to fabricate artificial reference images that corrupt benchmark interpretability.

---

## 9. Recommended Next Experiment (Section 13)

### **Recommendation: Structured Ingestion of Verified Real-CDR Lunar Altimetry (LOLA/SLDEM2015)**

**Do NOT attempt further optical or synthetic-proxy hillshade experiments until real DEM rasters are ingested.**

**Actionable Next Step**:
1. Use the existing [`LRODataHandler`](backend/data/lro_validator.py#L18) and [`lro_ode_client.py`](ML_model/lro_ode_client.py) infrastructure to ingest calibrated, real-world LOLA/SLDEM2015 elevation GeoTIFFs for `region_001` through `region_006`.
2. Populate exact solar elevation angles (`ohrc_sun_elevation_deg`, `tmc2_sun_elevation_deg`) in the region manifests directly from PDS4 XML labels.
3. Only after real, unmanipulated elevation rasters and true solar geometry are deposited in the repository, execute the hillshade synthesis and correspondence diagnostics.

---

## 10. Test Results & Production Integrity Confirmation

### 1. Test Suite Verification
- `python3 -m pytest -q`: **258 passed, 2 skipped** (100% pass rate).
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** (100% pass rate).

### 2. Model & Hyperparameter Freeze Confirmation
- `git diff -- ML_model/ai_verifier_model.pkl`: **Empty** (0 bytes modified).
- Production candidate matching defaults remain strictly frozen: `half_p=8`, `use_goa_preselection=False`.
- The 18 held-out groups were **never evaluated or accessed**.

### 3. Files Created / Modified
- Created: `evaluation/phase11_dem_hillshade_feasibility.md` (this report).
- Modified: None. Production code is strictly unchanged.
