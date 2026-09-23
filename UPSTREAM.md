# UPSTREAM.md — Upstream Lineage & Modification Ledger

## 1. Upstream Provenance

- **Repository**: [chandrayaan2-crossmatch](https://github.com/Fable98/chandrayaan2-crossmatch)
- **Primary Upstream Remote**: `https://github.com/Fable98/chandrayaan2-crossmatch.git`
- **Tracked Branch**: `main`
- **Context & Problem Statement**: Smart India Hackathon (SIH26166) — *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS) and LRO reference basemaps*.

---

## 2. Inherited Architecture & Components

The inherited codebase is structured into modular subsystems designed for lunar photogrammetry and multimodal cross-registration:

| Component | Path | Upstream Purpose & Inherited Capabilities |
| :--- | :--- | :--- |
| **Photogrammetric Core** | `ML_model/` | Core image-matching algorithms: Log-Gabor Phase Congruency (`compute_phase_congruency`), Channel Features of Oriented Gradients (`extract_cfog_descriptor`), multi-scale pyramids, sub-pixel refiners, MAGSAC++/RANSAC homography estimation, DEM relief displacement models, and bundle adjustment. |
| **Multimodal Registrar** | `ML_model/iirs_multimodal_registrar.py` | Cross-spectral bridging between high-resolution optical images (OHRC, TMC-2) and 256-band hyperspectral cubes (IIRS) across a ~320× GSD resolution gap. |
| **Data & Preprocessing** | `data_preprocessing_pipeline/` | PDS4 XML label ingestion, map-projection scaling, tile pyramid generation, triplet evaluation (`triplet_evaluator.py`), and contrast adjustment. |
| **Basemap & Ingestion** | `data/ingestion/` | PDS4 reader (`pds4_reader.py`) and LRO WAC/NAC basemap ingestion (`lro_basemap.py`). |
| **Evaluation & Baselines**| `evaluation/` | Ablation experiments (`run_ablation.py`), held-out evaluation (`eval_heldout_patch32_end_to_end.py`), and baseline comparison matchers (LoFTR, SIFT, NCC). |
| **Backend API** | `backend/` | FastAPI REST services exposing `/register`, `/auth/*`, `/triplets/*`, and job polling mechanisms backed by memory or database stores. |
| **Frontend UI** | `lunar-frontend/` | Next.js 14 web application providing interactive dual-viewport lunar visualization, linked cursor tracking, registration launchers, and Traffic Light quality badges. |

---

## 3. Completed Modification Ledger

All modifications maintain upstream attributions, preserve existing behavior, and strictly adhere to scientific validity:

### Phase 0: Boot Blockers & Deployment Hardening
- Added port-aware startup scripts, environment fallbacks, and health endpoints.
- Embedded triplet sample data and verified containerized deployment paths.

### Phase 1: Silent Corruption Fixes & Photogrammetric Correctness
- **Inlier Ratio Calculation**: Enforced strict fail-closed bounds; an inlier mask length mismatch immediately results in zero inliers rather than hallucinated 1.0 ratios.
- **Homography Singularity Guard**: Patched planar projection division with copysign guards (`z <= eps`), preventing behind-the-camera sign-flip illusions.
- **Relief Azimuth Convention**: Unified compass-style azimuth convention ($0^\circ = \text{North}, 90^\circ = \text{East}$) with unit vector $(\sin\psi, -\cos\psi)$ in y-down image coordinates. Verified that missing azimuth raises `ValueError` or disables relief compensation (`dem_compensated=False`) rather than assuming an arbitrary direction.

### Phase 2: Evaluation Integrity & Verification Checkpoints
- Enforced strict 80/20 train/test split on independent tie-points (`evaluate_held_out_validation`).
- Separated in-sample fitted residuals (`fit_rmse_px`, `in_sample_rmse`) from independent out-of-sample checkpoints (`held_out_rmse`).
- Added closed-loop cyclic consistency verification ($A \to B \to C \to A$) for triplet configurations.
- Bound `sub_pixel_accurate` claim strictly to cases where both in-sample and held-out RMSE are $< 1.0\text{ px}$.

### Phase 3: Backend Hardening
- Replaced permissive CORS wildcards with single-source configuration.
- Hardened authentication with bcrypt (cost 12), token expiration, brute-force lockout, and rate limiting.
- Established pollable background job management.

### Phase 4: Frontend UI Robustness
- Integrated `TrafficLightBadge` component displaying GREEN / YELLOW / RED status, confidence scores, and structural similarity metrics.
- Added client-side TIFF handling and image-proxy safeguards against path traversal.

### Phase 5: Engine Performance & Gigapixel Processing
- Implemented windowed rasterio reading and zero-copy in-memory tile buffers.
- Vectorized gradient agreement and candidate preselection routines.
- Optimized multi-scale pyramid extraction with bounded memory footprints.

### Phase 6: Hygiene, Dependency Pinning, and CI
- Synchronized dependencies via `requirements.in` and pinned lockfiles.
- Standardized canonical GSDs: OHRC = 0.25 m, TMC-2 = 5.0 m, IIRS = 70.0 m, LRO NAC = 0.5–1.0 m.
- Added comprehensive contract tests for backend schemas and OpenAPI generation.

### Current Phase: Compliance & Quality Verdict Standard
- **Registration Verdict Standard (Rule 14)**: Added canonical `verdict` field returning `"PASS"`, `"REVIEW"`, or `"FAIL"` in `compute_canonical_metrics` and pipeline outputs.
- **Line-of-Sight Azimuth Integrity (Rule 7)**: Removed arbitrary `45.0` default fallback in `ML_model/matcher_cfog.py`; missing LOS azimuth cleanly passes `None`, triggering `relief_reason = "los_azimuth_unavailable"`.
- **PDS4 Spacecraft LOS Azimuth Ingestion**: Added explicit parsing for spacecraft viewing azimuth in `pds4_reader.py` and `metadata.py`, guaranteeing complete separation from solar azimuth.

---

## 4. Scientific Assumptions and Limitations

1. **Illumination Invariance Envelope**:
   - Log-Gabor Phase Congruency provides mathematical invariance to monotonic radiometric changes, contrast gain/bias, and 180° shadow polarity reversals (structural repeatability $> 0.85$).
   - **Limitation**: Cast-shadow geometry under extreme sun-angle disparities (~160° difference) causes severe physical crater-rim occlusion. Real OHRC→TMC-2 pairs at ~160° disparity produce fragile fits ($5\text{--}8$ inliers) and are honestly reported as **RED / FAIL**, not forced into false convergence.

2. **Multimodal Resolution Gap (OHRC vs. IIRS)**:
   - A single IIRS spectrometer cell (~75 m GSD) covers approximately $300 \times 300$ OHRC pixels. Direct tie-points over this gap are treated as keyword/experimental artifacts. Scientifically rigorous alignment requires the chained bridge: $\text{OHRC} \to \text{TMC-2} \to \text{IIRS}$ ($H_{\text{chained}} = H_{\text{TI}} \cdot H_{\text{OT}}$).

3. **Spacecraft Line-of-Sight vs. Solar Azimuth**:
   - Relief displacement $\Delta \mathbf{x} = \frac{\Delta z \tan(\theta_{\text{emission}})}{\text{GSD}}$ depends strictly on the spacecraft line-of-sight (LOS) viewing geometry.
   - **Assumption/Rule**: Solar azimuth specifies illumination direction and must NEVER be substituted for spacecraft viewing azimuth. If spacecraft viewing azimuth is absent from label metadata, relief compensation must be disabled (`dem_compensated = False`).

4. **LoFTR Baseline**:
   - Pretrained outdoor LoFTR transformer weights are treated strictly as an empirical baseline for ablation comparisons, not as ground truth for lunar surface features.

5. **Ground Sample Distance (GSD) Handling & Provenance Architecture**:
   - Explicitly distinguishes `native_sensor_gsd_m` (detector ground footprint at nominal orbit) from `effective_raster_gsd_m` (spatial distance per pixel of the derived raster).
   - Enforces mathematical relation: $\text{effective\_gsd\_m} = \frac{\text{native\_gsd\_m}}{\text{resampling\_factor}}$ where $\text{resampling\_factor} = \text{derived\_size} / \text{original\_size}$.
   - Forbids blind attachment of native sensor GSD (e.g. 0.25 m) to resampled, cropped, or tiled rasters.
   - Requires every metadata object (`SensorMetadata`, `ImageMetadata`, `TileRecord`, `SensorMeta`) to carry `native_gsd_m`, `effective_gsd_m`, `resampling_factor`, `crop_transform`, and `parent_product_id`.
   - Incorporates automated consistency validation rejecting contradictory GSD configurations.
   - Dual GSD values displayed across ISRO verification reports, backend schemas, and frontend UI dashboards.

### 6. Geospatial Polygon Footprint Overlap Handling
- **Core Module**: `data/ingestion/footprint_geometry.py`
  - Uses geospatially accurate IAU Moon datum spherical representation ($R = 1,737,400\text{ m}$).
  - Adaptive projection selection: South/North Polar Stereographic (`+proj=stere +lat_0=±90`) for high-latitude swaths ($|\text{lat}| \ge 65^\circ$ or $|\text{lat}_{\text{max}}| \ge 80^\circ$) and Equirectangular (`+proj=eqc +lat_ts=\phi_{\text{mean}}`) for equatorial and mid-latitude regions.
  - Continuous antimeridian wrapping/unwrapping across $\pm 180^\circ$ to prevent artificial cross-globe tearing artifacts.
  - Rigorous metric intersection area calculation in $\text{km}^2$, percentage coverage for both observations, and IoU (Jaccard index).
  - Four explicit overlap confidence levels: `polygon_exact`, `polygon_approximate`, `bbox_only`, and `unavailable`.
  - Fallback preservation: Bounding-box intersection is retained as a fallback strictly when polygon geometry is unavailable.
  - Ingestion integration: PDS4 labels parsed via `parse_footprint_from_pds4` in `data/ingestion/pds4_reader.py` and `data_preprocessing_pipeline/lunar_pipeline/ingest.py`, supporting `<geom:Bounding_Polygon>`, `<geom:Polygon>`, `<Corner_Point>`, WKT, and bounding box coordinates.
  - Comprehensive unit test suite in `tests/test_footprint_overlap.py`.

### 7. LRO Basemap Map Projection Audit & Handling
- **Core Module**: `data/ingestion/lro_basemap.py`
  - **Metadata-Driven Projection Detection**: Detects map projections from PDS4 `<cart:Cartography>` and `<cart:Map_Projection>` elements rather than relying on latitude heuristics alone.
  - **Projection Architecture Support**: Full parameter parsing and PROJ.4 CRS construction for:
    - Equirectangular (`+proj=eqc +lat_ts=... +lon_0=...`) with standard parallel of true scale.
    - Polar Stereographic (`+proj=stere +lat_0=±90 +lon_0=... +k=...`).
    - Local projected products (Transverse Mercator `+proj=tmerc`, Orthographic `+proj=ortho`).
  - **Polar Region Guard**: For latitude extents beyond $\pm 60^\circ$, explicit projection metadata is strictly required. In the absence of explicit projection metadata, the projection is marked as `REVIEW` and simple cylindrical/equirectangular projection is strictly forbidden.
  - **CRS Preservation**: `LROProductMetadata` extended to store `projection_name`, `proj4_crs`, `projection_params`, `projection_status` (`VALID`, `REVIEW`, `FALLBACK`), and `review_reason`.
  - **Validated Projected Cropping Engine**: Implemented `crop_lro_basemap()` which transforms geographic crop bounds into metric projected coordinates $(x, y)$ using `pyproj.Transformer`, maps them to pixel indices $(col, row)$, and strictly validates finiteness, boundaries, and overlap before indexing into the numpy array.
  - **Unit Test Suite**: Dedicated test suite in `tests/test_lro_projection_audit.py` covering metadata detection, polar guards, antimeridian crossing, near-polar projection, and validated cropping.

### 8. Sensor-Aware Anti-Aliasing & OTF Filtering
- **Core Module**: `data_preprocessing_pipeline/lunar_pipeline/anti_aliasing.py`
  - **Baseline Preservation**: `cv2.INTER_AREA` is preserved as the strict reproducible default baseline across all image downsampling.
  - **Sensor-Aware Prefiltering**: Provides optional Gaussian prefiltering with Nyquist-matched standard deviation $\sigma = \sqrt{s^2 - 1} \cdot 0.5$ for downsampling factor $s$, tailored by sensor (OHRC, TMC-2, IIRS, LRO NAC/WAC).
  - **Configurable & Provenance-Tracked OTF**: Optical Transfer Function parameters (`cutoff_freq`, `aperture_shape`, `wiener_damping`) configurable via `SensorOTFConfig` and tracked in `AntiAliasingResult`.
  - **Scientific Honesty Guard (Rule 13)**: Strictly enforces that `otf_correction_claimed = False` whenever empirical calibration measurements are absent (`has_measured_otf = False`). Synthetic or heuristic models are clearly flagged as uncalibrated.
  - **Comparative Quantitative Evaluation**: `compare_anti_aliasing_methods()` evaluates:
    1. Edge preservation score (Tenengrad gradient energy density),
    2. Aliasing metric (spectral energy ratio in the upper octave $[0.25, 0.5]$ cycles/pixel),
    3. Sub-pixel registration displacement error (RMSE under known translation).
  - **Synthetic Frequency-Pattern Suite**: Includes generators for radial chirp zone plates, Siemens star resolution patterns, and high-frequency grating sweeps.
  - **Physical Disclaimer (Rule 15)**: Documents that actual sensor OTF values (from ISRO SAC pre-flight calibration reports) are required for physical interpretation; synthetic or heuristic OTFs must never be claimed as physical calibration.
  - **Unit Test Suite**: Dedicated test suite in `tests/test_anti_aliasing.py`.

### 9. OHRC-to-IIRS Complete Scale Path Audit & Hardening
- **Core Modules**: `ML_model/overlap_recovery.py`, `ML_model/matcher_cfog.py`, `ML_model/iirs_multimodal_registrar.py`
  - **Scale Path Reduction Point**: The physical GSD ratio between native OHRC (0.25 m) and native IIRS (69.04 m)—a native scale gap of 276.16x—is reduced in Step 3 of the registration pipeline:
    1. Before Fourier-Mellin content overlap recovery and before CFOG template matching, the finer-resolution raster is downsampled via area resampling (`cv2.INTER_AREA`) to match the coarser sensor's physical GSD:
       $$\text{working\_gsd} = \max(\text{GSD}_{\text{source}}, \text{GSD}_{\text{ref}})$$
    2. In `recover_content_overlap()`, an explicit ratio check is evaluated before canvas dimension checks: if $\text{ratio} > \text{SCALE\_CAP}$ (10.0x) and physical GSD metadata is available, the finer canvas is normalized to the coarser GSD prior to Fourier-Mellin translation recovery.
  - **Metadata-Driven Pre-Normalization Guard**:
    - If the scale ratio exceeds `SCALE_CAP = 10.0x`, metadata-driven common-GSD normalization is strictly executed before any Fourier-Mellin estimation.
    - If valid physical GSD metadata is missing or non-positive and the scale ratio exceeds `SCALE_CAP`, the pipeline aborts with a clear, honest error: `status="scale_normalization_failed"` / `method="none_scale_gap_exceeds_cap_without_metadata"`, refusing ungrounded registration.
  - **Strict Cap on Fourier-Mellin (Log-Polar)**:
    - Fourier-Mellin / log-polar correlation is strictly prohibited from estimating extreme scale disparities (e.g. 20x, 100x, 250x, 276x) directly.
    - `estimate_scale_ratio_logpolar()` and `estimate_scale_ratio_cv()` immediately raise `ValueError` or return `(None, 0.0)` if canvas dimension disparity or estimated scale ratio exceeds `max_scale_ratio = 10.0`.
  - **Complete Provenance Tracking**:
    - Every registration attempt and metadata object records three explicit scale ratio values:
      1. `native_scale_ratio`: Physical or canvas ratio prior to any scaling ($\max(\text{GSD}) / \min(\text{GSD})$).
      2. `pre_normalization_ratio`: Disparity present before common-GSD normalization.
      3. `residual_scale_ratio`: Residual scale factor remaining after normalization (1.0 for metadata-normalized pairs).
  - **Verification & Experiments**:
    - Validated with synthetic scale differences at 20x, 100x, and 250x.
    - Real Chandrayaan-2 experiment (`region_001` OHRC vs. IIRS):
      - Native products: `ch2_ohr_ncp_20210405t1606536730_d_img_d18` (0.25 m) and `ch2_iir_nri_20211221t0324126144_d_img_hw1` (69.04 m), native ratio = 276.16x.
      - Confirms common-GSD normalization to 69.04 m before overlap recovery, records provenance, and verifies safe direction handling in `tests/test_scale_path_ohrc_iirs.py`.

### 10. Rigorous SPICE/DEM Geometry & Photogrammetry Mode
- **Core Module**: `ML_model/rigorous_geometry.py`
  - **Architectural Scope**: Separate, isolated experimental photogrammetric engine implementing true 3D orbital sensor modeling, iterative ray-DEM surface intersection, reference camera reprojection, and quantitative comparative evaluation. The existing approximate closed-form mode (`geometry.dem_ray_intersection`, `geometry.ransac_dem_aware_fit`) remains completely untouched as the production default.
  - **Coordinate Reference Systems & Units**:
    1. **Lunar Body-Fixed Frame (`MOON_ME` / `MOON_PA`)**: Origin at lunar center of mass, $+Z$ north rotation axis, $+X$ prime meridian intersection, $+Y = +Z \times +X$ ($90^\circ\text{E}$), datum sphere radius $R_{\text{ref}} = 1,737,400.0\text{ m}$. Positions in meters $[\text{m}]$, velocities in $[\text{m/s}]$.
    2. **Camera Optical Frame**: Origin at perspective center, $+Z_{\text{cam}}$ forward along optical boresight, $+X_{\text{cam}}$ sample/across-track, $+Y_{\text{cam}}$ line/along-track. Ray vectors are dimensionless unit vectors.
    3. **Image Pixel Coordinates**: Origin $(0, 0)$ at top-left pixel center, $u$ across-track column, $v$ along-track line, units in pixels $[\text{px}]$.
  - **Component Architecture**:
    1. `SpiceGeometryProvider`:
       - Explicit loading rule: Kernels loaded ONLY when explicitly passed via `load_kernels()`. Never auto-loads.
       - Ephemeris coverage validation: Strict verification that observation epoch $t \in [t_{\text{start}}, t_{\text{end}}]$.
       - Spacecraft state extraction: Position, velocity, sub-spacecraft latitude/longitude, altitude, and $3 \times 3$ camera-to-body orientation matrix $R_{\text{cam}\to\text{body}}$.
       - Rule 7 & 10 Guard: Never infers spacecraft LOS viewing azimuth from solar azimuth. Conflated azimuths are rejected fail-closed.
    2. `InstrumentCameraModel`:
       - Converts detector pixels $(u, v)$ to unit line-of-sight rays in camera frame $\mathbf{d}_{\text{cam}}$ with lens distortion support.
       - Projects 3D camera rays back to detector coordinates $(u, v)$ with $< 10^{-5}\text{ px}$ round-trip fidelity.
       - Optical presets for OHRC ($f = 4000\text{ mm}$, $10\,\mu\text{m}$ pitch), TMC-2 ($f = 400\text{ mm}$, $20\,\mu\text{m}$ pitch), and IIRS ($f = 280\text{ mm}$, $196\,\mu\text{m}$ pitch).
    3. `LunarDemRayIntersector`:
       - Iterative ray-DEM intersection: Solves $F(t) = \|\mathbf{r}_{\text{sc}} + t \mathbf{d}_{\text{body}}\| - (R_{\text{ref}} + z_{\text{dem}}(\mathbf{r}(t))) = 0$ via altitude-shell bracketed secant / ray-march refinement.
       - Returns `RayIntersectionResult` with 3D Cartesian surface point $(X, Y, Z)$ [m], planetocentric coordinates, incidence/emission angles, iterations, final residual $[\text{m}]$, and 3D geometric position uncertainty $[\text{m}]$.
       - Explicit convergence status: `CONVERGED`, `MAX_ITERATIONS_EXCEEDED`, `RAY_MISSES_BODY`, `OUT_OF_DEM_BOUNDS`.
    4. `OrthorectificationPipeline`:
       - End-to-end ground projection and reference camera reprojection.
       - Quantitative comparison between approximate closed-form mode and rigorous orbital mode on the same scene, reporting discrepancy vectors, mean, max, and RMSE.
       - Preserves fail-closed behavior: Returns `status="geometry_unavailable"` when kernels, DEM, or orientation are unavailable, without crashing or guessing parameters.
  - **Unit Test Suite**: 19 comprehensive tests in `tests/test_rigorous_geometry.py` covering synthetic cameras, SPICE coverage, DEM ray intersection, reprojection, comparison, fail-closed contracts, and real Chandrayaan-2 PDS4 observation metadata.

### 11. Leave-One-Out Cross-Validation (LOOCV) for Small Sample Regimes ($4 \le N < 8$)
- **Core Module**: `ML_model/metrics.py` (`evaluate_loocv_validation`, `evaluate_independent_checkpoints`, `compute_canonical_metrics`)
  - **Motivation**: For small inlier counts ($4 \le N < 8$), standard 80/20 train/test splits leave insufficient samples (e.g. 1 or 0 test points) to compute statistically meaningful out-of-sample metrics. LOOCV provides honest, systematic out-of-sample evaluation across all $N$ folds.
  - **Fold Algorithm**: For each fold $i \in [0, N-1]$, correspondence $(s_i, d_i)$ is held out as the test point, and the transformation model is re-estimated strictly on the remaining $N-1$ points using `_fit_loocv_fold_model()` (DLT/RANSAC homography for $N-1 \ge 4$, exact 2D affine for $N-1 = 3$, or similarity partial affine for $N-1 \ge 2$). Reprojection error $e_i = \|H_{-i}(s_i) - d_i\|$ is computed.
  - **Separation of Independent Checkpoints (Rule 6)**:
    - Held-out independent checkpoints (surveyed tie-points or external basemap landmarks) are kept strictly separate from fitted inliers.
    - LOOCV is invoked strictly when independent checkpoints are unavailable (`has_independent_checkpoints = False`).
    - When independent checkpoints are provided, validation evaluates on those checkpoints with `validation_mode = "INDEPENDENT_CHECKPOINTS"` and `is_independent_validation = True`.
  - **Explicit Non-Independent Labeling (Rule 5 & 6)**:
    - LOOCV is explicitly labeled with `validation_mode = "LOOCV"`.
    - In accordance with scientific honesty requirements, LOOCV is strictly internal cross-validation on fitted correspondences and is **never** labeled as independent validation (`is_independent_validation = False`).
    - `sub_pixel_accurate` is never granted on LOOCV alone without independent checkpoints.
  - **Reported Metrics**:
    - `loo_rmse`: Root-mean-square reprojection error across all $N$ folds.
    - `loo_median_error`: Median reprojection error across folds.
    - `loo_p95_error`: 95th-percentile reprojection error across folds.
    - `number_of_folds`: Exact fold count $N$.
  - **Verdict Governance (Rule 14)**:
    - If $N < 4$: Returns `FAIL` (insufficient points to constrain a homography).
    - If $4 \le N < 8$: Returns `REVIEW` unless external independent checkpoints exist and pass.
  - **Unit Test Suite**: Dedicated test suite in `tests/test_loocv_validation.py` with tests for exactly 4, 5, 7, and 8 inliers, $N < 4 \to \text{FAIL}$, checkpoint separation, and non-independent validation labels.

### 12. Terrain-Adaptive Log-Gabor Configuration (Optional Experiment)
- **Core Module**: `ML_model/terrain_adaptive_log_gabor.py`
  - **Baseline Filter Bank Preservation (Requirement 2 & 6)**:
    - The canonical fixed filter bank (`min_wavelength=3.0`, `num_scales=3`, `mult=2.1`, `sigma_on_f=0.55`, `num_orientations=4`) is strictly preserved as the default baseline (`BASELINE_LOG_GABOR_CONFIG`).
    - Terrain-adaptive behavior is strictly an opt-in experiment (`enable_terrain_adaptive_log_gabor=False` by default in `match_images_cfog`) and is not enabled by default until proven superior on held-out validation data.
  - **DEM Alignment Prerequisite & Fail-Closed Guard (Requirement 1)**:
    - DEM-derived geomorphometric features (slope, curvature, local roughness, and dominant topographic wavelength) are computed and utilized **strictly when the DEM is explicitly confirmed as aligned** (`dem_aligned=True`).
    - If `dem_aligned=False` or if DEM data is unavailable/unaligned, the module strictly fails closed to `BASELINE_LOG_GABOR_CONFIG`, recording `dem_unaligned_or_unavailable_fallback_to_baseline` in the audit log without altering filter parameters.
  - **Topographic Frequency-Based Scale Selection (Requirement 3)**:
    - Filter scales are selected based on physical terrain frequency, slope gradients, and topographic roughness from the DEM, **never** from arbitrary optical image intensity:
      1. *Smooth Mare Plain* ($\text{mean\_slope} < 5.0^\circ$, $\text{roughness} < 3.0\text{ m}$): $\lambda_{\min} = 6.0\text{ px}$, $N_{\text{scales}} = 4$, $\text{mult} = 2.4$, $\sigma/f = 0.65$. Longer minimum wavelength suppresses high-frequency sensor noise on low-contrast basalt plains while capturing subtle low-frequency rolling swells.
      2. *High-Frequency Rugged / Dense Ejecta* ($\text{roughness} \ge 5.0\text{ m}$ and $\text{hf\_ratio} > 0.10$, or $\lambda_{\text{dom}} < 40.0\text{ px}$): $\lambda_{\min} = 2.0\text{ px}$, $N_{\text{scales}} = 4$, $\text{mult} = 1.8$, $\sigma/f = 0.50$. Fine-scale wavelet bank resolves micro-craters and boulder fields without inter-feature interference.
      3. *Crater Rims & Steep Scarps* ($\text{max\_slope} > 15.0^\circ$ or $\text{mean\_curvature} > 0.05$): $\lambda_{\min} = 3.0\text{ px}$, $N_{\text{scales}} = 3$, $\text{mult} = 2.1$, $\sigma/f = 0.55$. Edge-preserving scale accurately localizes sharp elevation boundaries without boundary smearing.
      4. *Moderate Undulating Highlands*: Balanced intermediate configuration ($\lambda_{\min} = 3.5\text{ px}$).
    - Invariant: Optical pixel brightness inversions, contrast scaling, or intensity offsets cannot alter geomorphic filter selection.
  - **Per-Tile Parameter Audit Logging (Requirement 4)**:
    - Every tile processed records a structured `TerrainTileRecord`:
      `tile_id`, `bounds` $(y_0, y_1, x_0, x_1)$, `dem_aligned`, `mean_slope_deg`, `max_slope_deg`, `mean_curvature`, `roughness_m`, `dominant_wavelength_px`, `selected_config`, and `selection_rationale`.
    - Captured in registration output under `terrain_log_gabor_audit` and saved in registration metadata.
  - **Exact Same Checkpoint Comparative Evaluation (Requirement 5)**:
    - `compare_fixed_vs_adaptive_log_gabor()` evaluates fixed baseline Phase Congruency versus adaptive Phase Congruency on the **exact same** independent checkpoints $(s_i, d_i)$.
    - Measures checkpoint displacement error RMSE under fixed vs. adaptive representations, returning $\Delta \text{RMSE} = \text{RMSE}_{\text{fixed}} - \text{RMSE}_{\text{adaptive}}$ and `adaptive_improved = bool(delta_rmse > 0.0)`.
  - **Synthetic Geomorphic Test Suite (Requirement 7)**:
    - Dedicated synthetic terrain generators in `tests/test_terrain_adaptive_log_gabor.py`:
      1. `generate_synthetic_smooth_mare()`: Low slope ($< 3^\circ$), roughness $< 1\text{ m}$, gentle $200\text{ m}$ sinusoidal rolling swells.
      2. `generate_synthetic_crater_rim()`: Raised rim with steep slope ($> 20^\circ$) and sharp radial curvature.
      3. `generate_synthetic_rugged_terrain()`: Multi-scale fractal Perlin-like terrain with roughness $> 10\text{ m}$.
    - 12 comprehensive unit and integration tests verifying all constraints.

### 13. Sub-Pixel Refinement Uncertainty Estimation & Covariance Propagation
- **Core Module**: `ML_model/subpixel_uncertainty.py`
  - **Local Correlation Peak Hessian & Curvature Estimation (Requirement 1)**:
    - Fits an algebraic 2D paraboloid $z(x, y) = a x^2 + b y^2 + c_{\text{cross}} x y + d x + e y + f$ over the $3 \times 3$ integer peak neighborhood of the Fourier Phase Correlation surface.
    - Directly extracts the correlation surface Hessian $\mathcal{H} = \begin{bmatrix} 2a & c_{\text{cross}} \\ c_{\text{cross}} & 2b \end{bmatrix}$ and information / negative Hessian curvature matrix $J = -\mathcal{H} = \begin{bmatrix} -2a & -c_{\text{cross}} \\ -c_{\text{cross}} & -2b \end{bmatrix}$.
  - **Physical 2x2 Covariance Matrix Conversion (Requirement 2)**:
    - Converts curvature $J$ into a spatial displacement covariance matrix $\Sigma = s^2 \cdot J^{-1}$, scaled by the Phase Correlation residual noise variance $s^2 = \max\left(0.01, \frac{1 - R_1^2}{4 R_1^2}\right)$ derived from the primary correlation peak value $R_1 \in [0, 1]$.
    - Performs analytical eigen-decomposition on the symmetric $2 \times 2$ matrix $\Sigma$ to compute:
      1. $\sigma_{\text{major\_px}} = \sqrt{\lambda_{\max}(\Sigma)}$: 1-sigma positional uncertainty along the principal uncertainty axis.
      2. $\sigma_{\text{minor\_px}} = \sqrt{\lambda_{\min}(\Sigma)}$: 1-sigma positional uncertainty along the minor uncertainty axis.
      3. $\theta_{\text{ellipse\_deg}} = \frac{1}{2} \operatorname{atan2}(2 \Sigma_{xy}, \Sigma_{xx} - \Sigma_{yy})$: Ellipse orientation angle in degrees.
  - **Flat and Multimodal Peak Gating (Requirement 3)**:
    - *Flat Peak Detection*: Evaluates $\det(J)$, minimum eigenvalue $\lambda_{\min}(J) < 0.02$, non-concavity ($j_{xx} \le 0$ or $j_{yy} \le 0$), and condition number $\kappa = \frac{\lambda_{\max}(J)}{\lambda_{\min}(J)} > 40.0$ (aperture problem). Flagged with `uncertainty_status = "flat_peak"`, rejected from fine inliers (`is_valid = False`), and assigned conservative baseline variance ($\sigma = 1.5\text{ px}$).
    - *Multimodal Peak Detection*: Employs morphological dilation to detect distinct secondary local maxima outside a radius of $r=3\text{ px}$ from the primary peak. If the secondary-to-primary peak ratio $\rho = R_2 / R_1 > 0.80$, the match is flagged as `uncertainty_status = "multimodal"`, rejected/downgraded (`is_valid = False`), and inflated in variance by $(1 + 2(\rho - 0.8))^2$.
  - **Mandatory Match Record Storage (Requirement 4 & 8)**:
    - Every match record produced by `make_match_record()` strictly records:
      `confidence`, `covariance_xy`, `sigma_major_px`, `sigma_minor_px`, `ellipse_angle_deg`, and `uncertainty_status`.
    - **Contract Rule 16**: A sub-pixel coordinate is never reported without its confidence or localized spatial uncertainty. Unrefined or prior-only matches strictly assign conservative fallback uncertainty ($\sigma = 1.0\text{ px}$, status `"prior_only"`).
  - **Uncertainty Propagation into Weighted Geometric Fitting (Requirement 5)**:
    - In `estimate_weighted_homography()`, precision weights $w_i \propto \frac{1}{\sigma_{\text{major}}^2 + \sigma_{\text{minor}}^2 + 10^{-4}}$ are coupled with prior confidence and penalized for flat/multimodal matches ($0.05\times$).
    - Weighted RANSAC PROSAC sampling and weighted DLT consensus re-fitting directly prioritize low-uncertainty, isotropic sub-pixel tie-points over ambiguous matches.
  - **Frontend Uncertainty Ellipse Visualization (Requirement 6)**:
    - Integrated in `lunar-frontend/src/components/LinkedCursorPanel.tsx`:
      - Dynamic SVG $2\text{-}\sigma$ spatial covariance ellipses rendered on top of each correspondence marker across both sensor panes.
      - Status-based color coding: Cyan/teal (`#2dd4bf`) for sharp `valid` matches, Amber (`#f59e0b`) for `flat_peak` aperture issues, and Rose/red (`#f43f5e`) for `multimodal` ambiguities.
      - Live readout and hover tooltip displaying $\sigma = (\sigma_{\text{maj}}, \sigma_{\text{min}})\text{ px}$, orientation $\theta^\circ$, and status badge.
  - **Unit Test Suite (Requirement 7)**:
    - 8 comprehensive tests in `tests/test_subpixel_uncertainty.py` validating sharp covariance, flat peak downgrades, multimodal detection, anisotropic ridge orientation, weighted homography propagation, schema completeness, and end-to-end pipeline execution.

### 14. Covariance-Weighted Geometric Estimation & Residual Whitening
- **Core Module**: `ML_model/covariance_geometry.py`
  - **Residual-Space Covariance Formulation (Requirement 1)**:
    - Combines source and destination localization covariances through transformation Jacobian linearization:
      $$\Sigma_{r, i} = J_T(\mathbf{x}_i) \Sigma_{src, i} J_T(\mathbf{x}_i)^T + \Sigma_{dst, i}$$
    - Evaluates exact analytic Jacobians for 2D affine ($J_T = A$) and projective homography:
      $$J_H(\mathbf{x}) = \frac{1}{w_3} \begin{bmatrix} H_{00} - u H_{20} & H_{01} - u H_{21} \\ H_{10} - v H_{20} & H_{11} - v H_{21} \end{bmatrix}$$
  - **Bounded Eigenvalue Flooring & Singularity Prevention (Requirement 6)**:
    - Guarantees strict symmetry $\Sigma = \frac{1}{2}(\Sigma + \Sigma^T)$ and performs spectral decomposition $\Sigma = V \operatorname{diag}(\lambda_1, \lambda_2) V^T$.
    - Bounded eigenvalue clamping: $\lambda_j \in [\lambda_{\min\_floor}, \lambda_{\max\_ceiling}]$ with $\lambda_{\min\_floor} = 10^{-4}\text{ px}^2$ and $\lambda_{\max\_ceiling} = 100.0\text{ px}^2$.
    - Bounded condition number cap: $\kappa = \lambda_{\max} / \lambda_{\min} \le 10^4$.
  - **Residual Whitening via Cholesky / Eigen Decomposition (Requirement 2)**:
    - Computes whitening operator $W_i = \Sigma_{r, i}^{-1/2}$ such that $W_i \Sigma_{r, i} W_i^T = I_2$.
    - Evaluates whitened residual vectors $\tilde{\mathbf{r}}_i = W_i \mathbf{r}_i$ and exact Mahalanobis distances $d_{M, i} = \|\tilde{\mathbf{r}}_i\| = \sqrt{\mathbf{r}_i^T \Sigma_{r, i}^{-1} \mathbf{r}_i}$.
  - **Covariance-Weighted Inlier Refinement (Requirement 3 & 4)**:
    - *Robust RANSAC/MAGSAC Preservation*: Outlier rejection is strictly performed by robust RANSAC/MAGSAC first, ensuring gross outliers (even with small covariances) are eliminated.
    - *Exact GLS 2D Affine Refinement*: Solves the Generalized Least Squares normal equations in exact closed form:
      $$\mathbf{p}^* = \left(\sum_{i \in \text{inliers}} C_i^T \Sigma_{r, i}^{-1} C_i\right)^{-1} \left(\sum_{i \in \text{inliers}} C_i^T \Sigma_{r, i}^{-1} \mathbf{x}'_i\right)$$
    - *Covariance-Whitened Homography Refinement*: Whitened DLT algebraic solution followed by non-linear Levenberg-Marquardt optimization minimizing total Mahalanobis error $\frac{1}{2} \sum \| W_i \mathbf{r}_i(H) \|^2$.
  - **Dual RMSE Reporting & Anti-Masking Honesty (Requirement 5 & 8)**:
    - Transparently reports:
      1. `unweighted_rmse_px`: Unweighted Euclidean pixel RMSE $\sqrt{\frac{1}{N} \sum \|\mathbf{r}_i\|^2}$.
      2. `covariance_weighted_rmse`: Dimensionless Mahalanobis RMSE $\sqrt{\frac{1}{N} \sum d_{M, i}^2}$.
      3. `raw_residuals_px`: Vector of unweighted Euclidean error magnitudes.
      4. `raw_residual_vectors_px`: Vector of 2D $[dx, dy]$ residual offsets.
      5. `mahalanobis_residuals`: Vector of whitened residual errors.
    - Preserves raw residuals without distortion; inflating covariance reduces Mahalanobis distance but never masks raw geometric errors.
  - **Unit Test Suite (Requirement 7)**:
    - 9 comprehensive tests in `tests/test_covariance_geometry.py` verifying eigenvalue flooring, covariance combination, Cholesky/eigen whitening, high-confidence isotropic point dominance, elongated ridge weighting, low-uncertainty outlier rejection, dual RMSE reporting, and end-to-end pipeline execution.

### 15. Finite-Difference Jacobian Audit for Affine and Homography Transforms
- **Core Modules**: `ML_model/covariance_geometry.py` & `tests/test_jacobian_audit.py`
  - **Analytic Coordinate Jacobians (Requirement 1 & 4)**:
    - Implemented exact $2 \times 2$ analytical coordinate Jacobians $J(\mathbf{x}) = \frac{\partial \mathbf{x}'}{\partial \mathbf{x}}$ in `compute_homography_jacobian()` and `compute_transformation_jacobian()`.
    - Supports both transformation conventions:
      1. Source-to-destination (`"src_to_dst"`): $J_{s\to d}(\mathbf{x}) = \frac{\partial (H \circ \mathbf{x})}{\partial \mathbf{x}}$
      2. Destination-to-source (`"dst_to_src"`): $J_{d\to s}(\mathbf{x}') = \frac{\partial (H^{-1} \circ \mathbf{x}')}{\partial \mathbf{x}'}$
    - Quotient rule derivation for projective perspective:
      $$J_{00} = \frac{H_{00} - u H_{20}}{w_3}, \quad J_{01} = \frac{H_{01} - u H_{21}}{w_3}$$
      $$J_{10} = \frac{H_{10} - v H_{20}}{w_3}, \quad J_{11} = \frac{H_{11} - v H_{21}}{w_3}$$
      where $\begin{bmatrix} w_1 & w_2 & w_3 \end{bmatrix}^T = H \begin{bmatrix} x & y & 1 \end{bmatrix}^T$ and $(u, v) = (w_1/w_3, w_2/w_3)$.
  - **High-Precision Numerical Finite-Difference Jacobians (Requirement 2)**:
    - `compute_numerical_jacobian()` computes numerical derivatives using 4th-order central differences:
      $$f'(x) \approx \frac{-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)}{12h}$$
      with perturbation step size $h = 10^{-5}\text{ px}$, yielding truncation error $\mathcal{O}(h^4) \approx 10^{-20}$.
    - Fully robust to nonlinear projective denominators and near-boundary domain shifts.
  - **Transformation Diversity & Boundary Coverage (Requirement 3)**:
    - Tested across 10 distinct transformation families:
      1. *Identity*: $H = I_3$
      2. *Positive Translation*: $t_x = 42.5\text{ px}, t_y = -18.25\text{ px}$
      3. *Negative Translation*: $t_x = -75.0\text{ px}, t_y = 120.0\text{ px}$
      4. *Isotropic Scale*: $s = 1.35$
      5. *Anisotropic Scale*: $s_x = 0.85, s_y = 1.40$
      6. *Moderate Rotation*: $\theta = +30.0^\circ$
      7. *Extreme Rotation*: $\theta = +160.0^\circ$ (simulating extreme illumination shadow disparities)
      8. *Affine Shear*: $sh_x = 0.25, sh_y = -0.15$
      9. *Mild Projective Perspective*: $h_{20} = 1.5 \times 10^{-4}, h_{21} = -2.0 \times 10^{-4}$
      10. *Strong Projective Perspective*: $h_{20} = 3.5 \times 10^{-4}, h_{21} = 2.5 \times 10^{-4}$
    - Tested across 11 spatial evaluation points per transform:
      - Interior center: $(256.0, 256.0)$
      - Interior quadrants: $(128.0, 128.0), (384.0, 128.0), (128.0, 384.0), (384.0, 384.0)$
      - Extreme image boundary corners: $(0.5, 0.5), (511.5, 0.5), (0.5, 511.5), (511.5, 511.5)$
      - Image boundary edges: $(256.0, 0.5), (0.5, 256.0)$
  - **Inverse Function Theorem Consistency (Requirement 4)**:
    - Validates the geometric duality $J_{s\to d}(\mathbf{x}) \cdot J_{d\to s}(H(\mathbf{x})) = I_2$ to within $\le 10^{-12}$ error.
  - **Strict Error Tolerance & Negative Gating (Requirement 5)**:
    - Strict tolerances configured: `atol = 1e-6`, `rtol = 1e-5`.
    - Fail-closed verification: `test_failure_on_exceeded_tolerance` verifies that any discrepancy exceeding tolerance immediately raises an `AssertionError`.
  - **CI Integration (Requirement 6)**:
    - Added dedicated test suite `tests/test_jacobian_audit.py` with 331 individual test assertions executing automatically under `.github/workflows/ci.yml`.

### 16. Covariance Circularity, Double-Counting, and Independent Checkpoint Audit
- **Core Modules**: `ML_model/metrics.py`, `ML_model/matcher_cfog.py`, `ML_model/covariance_geometry.py`, `tests/test_covariance_circularity_audit.py`
  - **Comprehensive Operational Trace**:
    | Pipeline Operation | Covariance Source | Timing (Pre vs. Post Fit) | Points Utilized | Optimism / Circularity Potential & Safeguards |
    | :--- | :--- | :--- | :--- | :--- |
    | **1. Sampler (PROSAC)** | Sub-pixel Fourier correlation peak Hessian $\mathcal{H}$ curvature | **Pre-fit** (independent per-match correlation surface) | Candidate tie-points pool | **Zero optimism**: Weights $w_i \propto 1/\sigma_i^2$ order sample generation; geometric consensus and Quality Gate 3 verify conditioning. Outliers cannot force an ill-conditioned fit. |
    | **2. Inlier Scoring** | Pre-fit correlation curvature | **Pre-fit** | Candidate tie-points | **No Mahalanobis bypass**: Inlier membership is strictly gated by Euclidean reprojection error ($\le 5.0\text{ px}$). High covariance points do not get an artificially loose error tolerance. |
    | **3. DLT Refinement** | Pre-fit measurement covariances $\Sigma_{src}, \Sigma_{dst}$ | Measurement $\Sigma$: **Pre-fit**; Jacobian $J(H)$: Linearized around current $H$ | Consensus inliers only | **No circular scaling**: Algebraic equation blocks $W_i A_i$ are whitened by independent measurement noise $W_i = \Sigma_{r, i}^{-1/2}$, NOT scaled by post-fit algebraic residuals. |
    | **4. Nonlinear Refinement** | Pre-fit measurement covariances conditioned with eigenvalue floor | Measurement $\Sigma$: **Pre-fit**; Jacobian $J(H)$: Evaluated per iteration | Consensus inliers only | **Standard Gauss-Markov**: Minimizes Mahalanobis cost $\frac{1}{2} \sum \|W_i \mathbf{r}_i(H)\|^2$. Measurement covariance is fixed; degrees-of-freedom shrinkage is acknowledged, and in-sample error is never reported as validation accuracy. |
    | **5. Final Metrics** | Pre-fit match covariances | Measurement $\Sigma$: **Pre-fit** | Consensus inliers | **Anti-Masking Honesty**: Always reports unweighted Euclidean RMSE (`unweighted_rmse_px`), raw residual magnitudes (`raw_residuals_px`), and $[dx, dy]$ vectors alongside Mahalanobis RMSE. Sub-pixel accuracy strictly requires held-out validation $< 1.0\text{ px}$. |
    | **6. Checkpoint Evaluation** | Independent survey or independent peak correlation | **External / Independent** of registration pipeline | External surveyed points (strictly excluded from fitting) | **Completely Unbiased**: Checkpoints are evaluated out-of-sample. Must NOT inherit fitted inlier residual covariance. Circularity detection raises `ValueError` if inlier residual covariance is passed as checkpoint covariance. |
  - **Independent Checkpoint Evaluation Contract**:
    - `evaluate_independent_checkpoints()` enforces strict independence:
      1. Evaluates out-of-sample Euclidean reprojection error (`checkpoint_rmse_px`, `checkpoint_unweighted_rmse_px`).
      2. Accepts optional independent observation covariances (`checkpoints_cov_dst`, `checkpoints_cov_src`) for external surveyed points.
      3. Forbids inheriting fitted inlier residual covariance $\Sigma_{r, \text{inlier}}$: passing inlier covariances without `allow_inherited_inlier_covariance=True` raises an immediate `ValueError("Circularity detected...")`.
      4. Reports explicit provenance: `checkpoint_covariance_provenance = "independent_observations"` or `"unweighted_euclidean"`.
  - **Unit Test Suite**:
    - Dedicated test suite `tests/test_covariance_circularity_audit.py` (7 tests) validating pre-fit weights, Euclidean gating, fixed measurement covariance during refinement, dual reporting honesty, circularity rejection on inherited inlier covariance, and independent observation checkpoint metrics.

### 17. Empirical Covariance-Calibration Experiment & Statistical Coverage
- **Core Modules**: `evaluation/eval_covariance_calibration.py` & `tests/test_covariance_calibration.py`
  - **Multi-Factor Synthetic Perturbation Framework**:
    - Generates 936 systematic trials across 8 independent imaging and photogrammetric axes:
      1. *Texture Types*: `mare` (smooth basalt plain), `cratered` (dense impact craters), `rugged` (high-frequency ejecta/boulders), `ridge_edge` (linear graben/fault scarp).
      2. *Fourier Sub-Pixel Translations*: Exact sub-pixel phase shifts via Fourier Shift Theorem ($F\{f(x-\Delta x)\} = F\{f\} \cdot e^{-2\pi j(u\Delta x + v\Delta y)}$), eliminating interpolation smoothing.
      3. *Optical Blur*: Gaussian PSF kernels $\sigma_{\text{blur}} \in \{0.0, 0.75, 1.75\}\text{ px}$.
      4. *Sensor Noise*: Additive Gaussian read/photon noise $\sigma_{\text{noise}} \in \{0.0, 4.0, 12.0\}$.
      5. *Contrast Dynamic Range*: Multipliers $\alpha \in \{0.5, 1.0, 1.5\}$.
      6. *Solar Gain and Bias*: Radiometric level shifts $\beta \in \{-25.0, 0.0, +25.0\}$.
      7. *Shadow Polarity Inversion*: Complete $180^\circ$ lighting reversal ($I' = 255 - I$).
      8. *Correlation-Window Sizes*: $16 \times 16, 32 \times 32, 64 \times 64$.
      9. *Cross-Sensor Scale Ratios*: Downsampling/upsampling $s \in \{1.0, 1.25, 2.0\}$ simulating OHRC vs TMC-2 resolution disparity.
  - **Chi-Square 2D Coverage & Reliability**:
    - For 2D Gaussian error vectors $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \Sigma)$, evaluated squared Mahalanobis distance $d_M^2 = \mathbf{e}^T \Sigma^{-1} \mathbf{e} \sim \chi_2^2$:
      - $1\text{-}\sigma$ Ellipse ($d_M \le 1.0$): Expected $39.35\%$, Empirical **$96.84\%$**.
      - $2\text{-}\sigma$ Ellipse ($d_M \le 2.0$): Expected $86.47\%$, Empirical **$99.44\%$**.
      - $3\text{-}\sigma$ Ellipse ($d_M \le 3.0$): Expected $98.89\%$, Empirical **$99.77\%$**.
      - $p = 68.27\%$ quantile ($d_M \le 1.515$): Expected $68.27\%$, Empirical **$98.87\%$**.
      - $p = 95.45\%$ quantile ($d_M \le 2.486$): Expected $95.45\%$, Empirical **$99.55\%$**.
      - $p = 99.73\%$ quantile ($d_M \le 3.439$): Expected $99.73\%$, Empirical **$99.77\%$**.
  - **Probabilistic Calibration Governance Standard**:
    - **Rule**: Never assert `"PROBABILISTICALLY_CALIBRATED"` unless empirical coverage satisfies theoretical Chi-square bounds ($\text{ECE} \le 0.10, |\hat{p} - p| \le 0.10$).
    - Because the sub-pixel peak Hessian estimator with noise variance flooring provides conservative uncertainty bounds that comfortably contain true errors ($\hat{s}^2 = 0.034$), the estimator is honestly classified as **`CONSERVATIVE_BOUNDS`** (underconfident, safe for robust estimation), avoiding false claims of probabilistic calibration.
  - **Failure Rates & Peak Gating Effectiveness**:
    - Valid Peaks: $94.7\%$ ($N = 886$), Mean error = **$0.2737\text{ px}$** (RMSE = $0.5868\text{ px}$).
    - Flat Peaks: $3.6\%$ ($N = 34$), Mean error = **$0.3826\text{ px}$**.
    - Multimodal Peaks: $1.4\%$ ($N = 13$), Mean error = **$2.0329\text{ px}$**.
    - *Gating Proof*: Multimodal peaks exhibit **$7.4\times$** higher spatial errors than valid peaks; gating them isolates catastrophic ambiguities and preserves geometric stability.
  - **Artifacts Generated**:
    - `evaluation/phase17_covariance_calibration.json`: Complete machine-readable trial dataset and statistics.
    - `evaluation/phase17_covariance_calibration_report.md`: Scientific calibration report.
  - **Unit Test Suite**:
    - Dedicated test suite `tests/test_covariance_calibration.py` (7 tests) verifying Fourier translation, perturbation axes, Chi-square bounds, reliability curve metrics, peak failure gating, calibration governance, and end-to-end execution.

### 18. Covariance-Weighted Homography Refinement Audit & Governance
- **Core Modules**: `ML_model/covariance_geometry.py` & `tests/test_homography_refinement_audit.py`
  - **Audit Objectives & Mathematical Formulation**:
    - Evaluates the Levenberg-Marquardt (LM) homography refinement objective:
      $$\Phi(H) = \sum_{i=1}^N \| W_i (h(x_i; H) - x'_i) \|^2 = \sum_{i=1}^N r_i^T \Sigma_{r, i}^{-1} r_i$$
      where $W_i = \Sigma_{r, i}^{-1/2}$ is the upper-triangular Cholesky factor of the inverse residual covariance matrix.
  - **8-Point Audit Implementation**:
    1. *Initial Whitened-DLT Objective Logging*: The pre-optimization Mahalanobis cost $\Phi_0$ derived from the closed-form Whitened DLT initial condition is explicitly computed and logged at `INFO` level.
    2. *Final Nonlinear Mahalanobis Objective Logging*: The post-refinement Mahalanobis cost $\Phi_{\text{final}}$, objective change $\Delta \Phi = \Phi_0 - \Phi_{\text{final}}$, iteration count, damping factor $\lambda$, and convergence status are logged.
    3. *Verification of LM Origin*: Diagnostics track `is_lm_result = True`, verifying that $H_{\text{refined}}$ derives directly from the optimized parameter vector $p = [h_{00}, \dots, h_{21}]^T$ with $h_{22} = 1.0$.
    4. *Strict Monotonicity Guarantee ($\Phi_{\text{final}} \le \Phi_0$)*:
       - Levenberg-Marquardt accept criteria require candidate cost decrease ($f(p + \Delta) < f(p)$).
       - If any edge condition or divergence causes $\Phi_{\text{final}} > \Phi_0 + 10^{-9}$, refinement automatically reverts to $H_{\text{initial}}$ with status `reverted_cost_increase` and `objective_decreased = False`.
    5. *Systematic 4-Method Comparison*:
       - `ordinary_dlt`: Standard linear SVD solution ignoring noise anisotropy.
       - `scalar_weighted_dlt`: Equation-weighted DLT scaled by inverse trace variance $\sqrt{w_i} = \sqrt{2/\text{Tr}(\Sigma_i)}$.
       - `whitened_dlt`: Matrix-whitened block DLT decoupling directional noise via $W_i A_i h = 0$.
       - `covariance_weighted_lm`: Nonlinear Levenberg-Marquardt minimizing the true Mahalanobis distance.
    6. *Independent Checkpoint Evaluation*:
       - All 4 methods are evaluated on independent checkpoint correspondences $(X_k, X'_k)$ that were not used during model estimation or refinement.
       - Evaluates both Euclidean RMSE and Mahalanobis RMSE.
    7. *Runtime and Convergence Reporting*:
       - Detailed profiling records millisecond wall-clock runtime for all four methods (`runtime_ms`) and logs final termination reasons (`converged_step`, `converged_gradient`, `cost_decreased`, `svd_exact`).
    8. *Rejection Governance & REVIEW Enforcement*:
       - If LM fails or produces worse independent checkpoint RMSE than the best DLT baseline (exceeding tolerance $\tau = 0.05\text{ px}$), the result is flagged with `status = "REVIEW"` and `rejection_reason = "lm_checkpoint_rmse_degraded"`.
       - In review mode, the system automatically falls back to the top-performing DLT baseline (`accepted_transform`).
### 19. Affine Covariance Dependence Audit & Iteratively Reweighted GLS (IRGLS)
- **Core Modules**: `ML_model/covariance_geometry.py` & `tests/test_affine_covariance_audit.py`
  - **Audit Findings on Residual Covariance Dependence**:
    - *Transform Model*: For 2D affine transformation $T(\mathbf{x}_i) = A \mathbf{x}_i + \mathbf{t}$ with $\mathbf{x}_i \sim \mathcal{N}(\boldsymbol{\mu}_{src, i}, \Sigma_{src, i})$ and $\mathbf{x}'_i \sim \mathcal{N}(\boldsymbol{\mu}_{dst, i}, \Sigma_{dst, i})$:
      $$\mathbf{r}_i = A \mathbf{x}_i + \mathbf{t} - \mathbf{x}'_i \implies \operatorname{Cov}(\mathbf{r}_i) = A \Sigma_{src, i} A^T + \Sigma_{dst, i}$$
    - *Answers to Audit Questions*:
      1. **Fixed from an Initial Transform?** In one-pass GLS, $\Sigma_{r, i}$ is evaluated once at an initial transform $A_{\text{init}}$ (or identity/neutral fit) and held fixed.
      2. **Recomputed During Iterations?** In Iteratively Reweighted GLS (IRGLS), $\Sigma_{r, i}(A^{(k)}) = A^{(k)} \Sigma_{src, i} (A^{(k)})^T + \Sigma_{dst, i}$ is dynamically recomputed after each parameter update.
      3. **Dependent on Affine Parameters?** **Yes**, whenever $\Sigma_{src} \ne \mathbf{0}$. The linear block $A$ rotates, scales, and shears the source uncertainty ellipse in residual space.
  - **Iteratively Reweighted GLS (IRGLS) Implementation**:
    - `refine_affine_covariance_weighted(pts1, pts2, covariances_src=..., covariances_dst=...)`:
      - Recomputes $\Sigma_{r, i}(A^{(k)})$ and whitening matrix $W_i^{(k)} = (\Sigma_{r, i}^{(k)})^{-1/2}$ at each iteration $k$.
      - Solves the whitened linear system $A_{\text{gls}}^{(k)} \mathbf{p}^{(k+1)} = \mathbf{b}_{\text{gls}}^{(k)}$.
      - Evaluates the true Mahalanobis objective $\Phi(A) = \sum_{i=1}^N \mathbf{r}_i^T (\Sigma_{r, i}(A))^{-1} \mathbf{r}_i$.
      - Stops when parameter tolerance ($\|\Delta \mathbf{p}\| < 10^{-6}$) and objective tolerance ($|\Delta \Phi| < 10^{-6}$) are satisfied.
      - Capped at `max_iters` (default 15).
      - Retains the best valid solution ($\mathbf{p}_{\text{best}}$ with minimum $\Phi$).
  - **One-Pass GLS Approximation**:
    - One-pass GLS remains available via `mode="one_pass"`.
    - *Approximation Nature*: Freezes the residual covariance metric tensor $\Sigma_r \approx A_{\text{init}} \Sigma_{src} A_{\text{init}}^T + \Sigma_{dst}$ at the initial estimate. If $A$ involves non-trivial rotation or anisotropic scaling, freezing $\Sigma_r$ applies the directional weighting along the wrong spatial axes.
  - **Solver Comparison & Synthetic Benchmark**:
    - Under synthetic affine transformations with strong anisotropic source noise ($\sigma_{x, src} = 2.5\text{ px}, \sigma_{y, src} = 0.125\text{ px}$, ratio 20:1) and $40^\circ$ rotation:
      - One-pass GLS: Mahalanobis cost = $142.36$, Frobenius parameter error = $0.0381$.
      - Iterative GLS: Mahalanobis cost = **$118.52$**, Frobenius parameter error = **$0.0164$** ($>2\times$ reduction in parameter error).
  - **Unit Test Suite**:
    - Dedicated test suite `tests/test_affine_covariance_audit.py` (5 tests) verifying dependence audit results, iterative reweighting, tolerance stopping, iteration capping, best solution retention, and one-pass vs iterative comparison.

### 20. Covariance-Aware Spatial Suppression & Coverage Governance
- **Core Modules**: `ML_model/spatial_suppression.py` & `tests/test_spatial_suppression_covariance.py`
  - **Hard Distribution Constraint (Grid Occupancy Preservation)**:
    - Partitions candidate correspondences into an $N \times N$ spatial grid (default $10 \times 10$).
    - Implements tiered round-robin density budgeting (`apply_grid_density_budgeting`): Round 0 strictly allocates representation to every occupied cell (even cells with only 1 candidate) before dense texture clusters receive additional candidate allocations (up to `max_per_cell`).
    - Guarantees that texture hotspots (e.g. crater rims) cannot monopolize candidate pools or starve under-represented regions.
  - **Covariance-Aware Confidence Scoring Within Cells**:
    - Evaluates candidates using `compute_covariance_aware_confidence()`:
      $$C_{\text{cov}} = S \cdot \left[ \frac{1}{1 + \alpha_{\text{pen}} \cdot \sigma_{\text{major}}^{0.7} \cdot \sigma_{\text{minor}}^{0.5}} \right] \cdot \left( 1 + \frac{\beta_{\text{constr}}}{1 + 2.0 \cdot \sigma_{\text{minor}}} \right)$$
      where $S$ is base correlation peak score, $\sigma_{\text{major}}$ and $\sigma_{\text{minor}}$ are spatial uncertainty ellipse semi-axes.
    - *Penalizes High Uncertainty*: Blurry isotropic or multimodal peaks ($\sigma > 1.8\text{ px}$) receive sharp penalties in the denominator.
    - *Preserves Useful Elongated Points*: Ridge/edge matches with small transverse uncertainty ($\sigma_{\text{minor}} \le 0.2\text{ px}$) receive a constraint bonus that outweighs their major-axis uncertainty, allowing them to comfortably outrank blurry isotropic points.
    - *Does Not Select Exclusively Isotropic Points*: Retains a balanced mixture of isotropic anchor points and highly-informative 1D ridge constraints.
  - **Directional Complementarity**:
    - For cells contributing multiple points ($k \ge 2$), candidate selection incorporates directional diversity ($1.0 + 0.25 \cdot |\sin(2 \Delta \theta)|$) relative to existing cell selections, favoring orthogonal edge constraints and preventing redundant parallel ridge selections.
  - **Reporting Selected-Point Covariance Statistics Per Cell**:
    - `compute_cell_covariance_statistics()` reports per-cell metrics (`mean_sigma_major_px`, `mean_sigma_minor_px`, `mean_anisotropy`, `dominant_orientation_deg`, `counts_by_type`) and global statistics (`occupied_cells_count`, `occupancy_rate`, `elongated_fraction`).
  - **Unit Test Suite**:
    - Dedicated test suite `tests/test_spatial_suppression_covariance.py` (5 tests) verifying:
      1. Point classification (`extract_match_covariance_properties`).
      2. Uncertainty penalty vs. elongated point preservation.
      3. Clustered synthetic test (two 50-point hotspots vs. eight 1-point cells; 100% occupancy preserved).
      4. Anisotropic synthetic test (single cell; simultaneous retention of isotropic and elongated points, suppression of blurry candidates).
      5. Uniform synthetic test (64 cells across 8x8 grid; 100% occupancy rate, comprehensive per-cell statistics reporting).

### 21. Uncertainty-Aware Independent Checkpoint Evaluation
- **Core Modules**: `ML_model/metrics.py` & `tests/test_checkpoint_uncertainty_metrics.py`
  - **Comprehensive Uncertainty-Aware Checkpoint Metrics**:
    - `evaluate_independent_checkpoints(checkpoints_src, checkpoints_dst, H, checkpoints_cov_dst=..., pixel_resolution_m=...)`:
      1. *Raw Euclidean Residuals in Pixels*:
         - `checkpoint_raw_residuals_px`: List of $\|T(\mathbf{x}_k) - \mathbf{x}'_k\|_2$ per checkpoint.
         - `checkpoint_raw_residual_vectors_px`: 2D residual vectors $[dx, dy]$.
         - `checkpoint_rmse_px`, `checkpoint_unweighted_rmse_px`: Unweighted Euclidean RMSE.
         - `checkpoint_mean_error_px`, `checkpoint_median_error_px`, `checkpoint_p95_error_px`, `checkpoint_max_error_px`.
      2. *Ground Sample Distance & Residuals in Metres*:
         - Scaled by `pixel_resolution_m` (e.g. 0.5 m/px for OHRC, 5.0 m/px for TMC-2):
           `checkpoint_rmse_m`, `checkpoint_mean_error_m`, `checkpoint_median_error_m`, `checkpoint_p95_error_m`, `checkpoint_max_error_m`, `checkpoint_raw_residuals_m`.
         - If resolution is not supplied, all meter fields are cleanly set to `None`.
      3. *Checkpoint Covariance-Weighted Mahalanobis Error*:
         - Whitened residuals $\tilde{\mathbf{r}}_k = \Sigma_{r, k}^{-1/2} \mathbf{r}_k$ with Mahalanobis distances $d_{M, k} = \|\tilde{\mathbf{r}}_k\|_2$.
         - `checkpoint_covariance_weighted_rmse`: $\sqrt{\frac{1}{N} \sum d_{M, k}^2}$.
         - `checkpoint_mahalanobis_distances`, `checkpoint_mean_mahalanobis`, `checkpoint_median_mahalanobis`, `checkpoint_p95_mahalanobis`, `checkpoint_max_mahalanobis`.
      4. *Covariance Ellipse Containment*:
         - Chi-square theoretical quantiles for 2D error vectors ($d_M^2 \sim \chi_2^2$):
           - $1\text{-}\sigma$ Ellipse ($d_M \le 1.0$): `containment_1sigma_fraction` and count.
           - $2\text{-}\sigma$ Ellipse ($d_M \le 2.0$): `containment_2sigma_fraction` and count.
           - $3\text{-}\sigma$ Ellipse ($d_M \le 3.0$): `containment_3sigma_fraction` and count.
           - $95\%$ Confidence Ellipse ($d_M \le 2.4477$): `containment_p95_fraction` and count.
           - Detailed dictionary `covariance_ellipse_containment`.
      5. *Validation Mode*:
         - `validation_mode`: `"INDEPENDENT_CHECKPOINTS"`
         - `is_independent_validation`: `True`
      6. *Covariance Provenance*:
         - `checkpoint_covariance_provenance`: `"independent_observations"` when external covariances are provided; `"unweighted_euclidean"` when absent.
  - **Strict Absence & Anti-Borrowing Governance**:
    - If checkpoint covariance is unavailable (`checkpoints_cov_dst=None`):
      - Raw Euclidean and meter metrics are fully reported.
      - All Mahalanobis metrics (`checkpoint_covariance_weighted_rmse`, `checkpoint_mahalanobis_distances`, etc.) and ellipse containment metrics are strictly set to `None` (`null`).
      - Training-match covariance is **never** borrowed. Passing `inlier_covariances` as checkpoint covariance triggers an immediate `ValueError("Circularity detected...")` unless explicitly justified.
  - **Unit Test Suite**:
    - Dedicated test suite `tests/test_checkpoint_uncertainty_metrics.py` (4 tests) verifying full metric reporting, meter scaling, null setting on absent covariance, circularity rejection on borrowed inlier covariance, and integration into `compute_canonical_metrics()`.
