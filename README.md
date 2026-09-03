# Chandrayaan-2 Multi-Modal Cross-Sensor Image Correspondence

### SIH Problem Statement 26166
**Title**: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)  
**Organization**: Indian Space Research Organisation (ISRO)

---

## 1. Executive Summary

This repository provides an open, reproducible, and photogrammetrically defensible pipeline for cross-sensor image correspondence between Chandrayaan-2 orbital instruments:
- **Orbiter High-Resolution Camera (OHRC)**: High-resolution panchromatic imaging (~0.25–0.32 m GSD).
- **Terrain Mapping Camera-2 (TMC-2)**: Stereo panchromatic triplets (~4–5 m GSD) supporting lunar surface topographic mapping.
- **Imaging Infrared Spectrometer (IIRS)**: Hyperspectral sensor (~70–80 m GSD) across 256 contiguous bands (~0.8–5.0 µm) providing mineralogical and volatile signatures.

### Primary Supported Scope
* **Primary Registration Pipeline**: High-precision correspondence between **OHRC and TMC-2** (~16–20× linear physical resolution difference).
* **IIRS Co-Registration Extension**: Co-registration of lower-resolution hyperspectral imagery as a spatial-spectral contextual overlay. IIRS is treated honestly as an ~70–80 m spectrometer product, without unphysical claims of sub-meter spatial reconstruction.
* **Baseline Alternative**: A pretrained LoFTR baseline is provided for comparative evaluation alongside the primary structural engine.

---

## 2. Scientific & Engineering Challenges

| Challenge | Physical Phenomenon | Engineering Solution in this Pipeline |
| :--- | :--- | :--- |
| **Physical Scale Disparity** | OHRC (~0.25 m) vs. TMC-2 (~5 m) is a ~20× linear resolution gap. Windowing identical pixel patches covers drastically different ground areas (32 m vs. 640 m). | **Common Physical-GSD Normalization**: Resamples source imagery to a shared working physical ground footprint (~5 m/px) prior to coarse matching and patch correlation. |
| **Solar Incidence & Illumination** | Drastic solar azimuth/elevation changes invert crater rim shadows, create false intensity gradients, and defeat raw pixel cross-correlation. | **Illumination-Robust Structural Representations**: 2D Log-Gabor Phase Congruency and CFOG (Channel Features of Oriented Gradients) extract frequency-phase edge features invariant to contrast reversals. |
| **Topographic Relief Displacement** | Off-nadir emission angles on 3D cratered lunar terrain induce parallax displacement proportional to terrain elevation. | **DEM Relief Displacement Compensation**: Ingests lunar DEM elevation and spacecraft viewing geometry to compensate local parallax shifts prior to matching. |
| **Evaluation Bias** | In-sample RANSAC RMSE evaluated strictly on the surviving inliers is self-fulfilling. | **Fit RMSE vs. Held-Out Validation RMSE**: Withholds a 20% held-out inlier split to evaluate out-of-sample reprojection error on verified correspondences. |
| **Correspondence Integrity** | Unreliable algorithms fabricate corner points when matching fails. | **Zero Synthetic Fallbacks**: Failed registrations cleanly report failure diagnostic statuses without fabricating correspondences or identity matrices. |

---

## 3. Sensor Specifications

| Sensor | Modality | Spectral Range | Nominal Spatial Resolution (GSD) | Mission Function |
| :--- | :--- | :--- | :--- | :--- |
| **OHRC** | Panchromatic optical | ~0.45–0.70 µm | ~0.25–0.32 m | Ultra-high resolution lunar lander site characterization and crater hazard assessment |
| **TMC-2** | Panchromatic optical stereo | ~0.40–0.85 µm | ~4–5 m | Stereo triplets (Fore, Nadir, Aft) for lunar 3D digital elevation model (DEM) generation |
| **IIRS** | Hyperspectral | ~0.80–5.00 µm | ~70–80 m | 256 contiguous spectral channels for mineralogical mapping and lunar hydration/OH detection |

*Note: Nominal spatial resolutions correspond to the ~100 km circular lunar polar orbit. Resampling multi-sensor imagery to a common processing grid normalizes pixel coordinates, but does not alter the underlying sensor resolution limit.*

---

## 4. End-to-End System Architecture

```text
Chandrayaan-2 Orbital Products (PDS4 XML/IMG, GeoTIFF, Cubes)
                        │
                        ▼
   Metadata & Observation Geometry Extraction
   (Sensor type, GSD, Solar Azimuth, Emission Angle, Provenance)
                        │
                        ▼
   Common Physical-GSD Normalization
   (Resamples OHRC to reference working GSD ~5 m/px with INTER_AREA)
                        │
                        ▼
   DEM Relief Displacement Compensation
   (Terrain elevation parallax correction under off-nadir geometry)
                        │
                        ▼
   Illumination-Robust Structural Feature Extraction
   (2D Log-Gabor Phase Congruency & CFOG gradient channel representations)
                        │
                        ▼
   Spatially Distributed Coarse Matching
   (Configurable NxN spatial grid binning with maximum matches per cell)
                        │
                        ▼
   Local Fourier Phase Correlation Sub-Pixel Refinement
   (2D Fourier cross-power with log-domain peak interpolation)
                        │
                        ▼
   Robust Geometric Estimation & Quality Gates
   (RANSAC projective/affine estimation with condition number & determinant sanity checks)
                        │
                        ▼
   Complete Registered Product Package Generation
   ┌────────────────────┬────────────────────┬────────────────────┐
   │ registered_source  │ checkerboard_qa    │ matches.json       │
   │ (.tif / GeoTIFF)   │ (.png 50px blocks) │ (native coords)    │
   ├────────────────────┼────────────────────┼────────────────────┤
   │ preview.png        │ metrics.json       │ metadata.json      │
   └────────────────────┴────────────────────┴────────────────────┘
                        │
                        ▼
   Canonical Quantitative Evaluation
   (Fit RMSE, Held-out Inlier Validation RMSE, Spatial Coverage, Uniformity Score)
                        │
                        ▼
   IIRS Hyperspectral Co-Registration & Spatial-Spectral Overlay
```

---

## 5. Quantitative Evaluation Metrics

All metrics in the repository are computed via a single canonical module ([`ML_model/metrics.py`](ML_model/metrics.py)):

1. **In-Sample Fit RMSE (`fit_rmse_px`)**:
   $$\text{RMSE}_{fit} = \sqrt{\frac{1}{N_{inliers}} \sum_{i=1}^{N_{inliers}} \|H p_i - q_i\|^2}$$
   Evaluated strictly on correspondences used to estimate the transformation matrix $H$.
2. **Held-Out Correspondence Validation RMSE (`validation_rmse_px`)**:
   Splits verified inlier points into training (80%) and validation (20%) sets. Re-estimates $H$ on training points and computes error exclusively on unseen held-out inlier points to measure transformation stability.
   *(Note: Synthetic benchmarks provide known ground-truth validation; real-image metrics provide held-out inlier correspondence validation).*
3. **Inlier Ratio (`inlier_ratio`)**:
   $$\text{Ratio} = \frac{N_{inliers}}{\max(1, N_{raw\_matches})}$$
4. **Spatial Coverage (`spatial_coverage`)**:
   $$\text{Coverage} = \frac{\text{Occupied Grid Cells}}{\text{Total Grid Cells (e.g. 100)}}$$
5. **Spatial Uniformity Score (`spatial_uniformity`)**:
   Combines grid coverage with match count dispersion: $\text{Score} = \text{Coverage} \times \exp(-0.3 \cdot \frac{\sigma}{\mu + \epsilon})$.
6. **Registration Quality Tiers**:
   - `FAILED`: Inliers < 4 or geometric conditioning gate failure.
   - `LOW_CONFIDENCE`: 4 to 9 inliers or spatial coverage < 10% (characteristic of small 512x512 demo crops).
   - `ACCEPTED`: 10 to 19 inliers and spatial coverage >= 10%.
   - `HIGH_CONFIDENCE`: >= 20 inliers, coverage >= 20%, and Fit RMSE < 2.0 px.
7. **Transformation Quality Gates**:
   Evaluates condition number, determinant positivity, and scale ratio to reject singular or severely distorted transformations.

---

## 6. Empirical Benchmark Results

### A. Synthetic Multi-Directional Sub-Pixel Displacement Benchmark
Evaluated across 240 trials spanning 8 directions (+X, -X, +Y, -Y, diag_+X_+Y, diag_+X_-Y, diag_-X_+Y, diag_-X_-Y), 6 fractional magnitudes (0.05, 0.10, 0.20, 0.30, 0.50, 0.75 px), and five separately seeded synthetic terrain realizations ([`scripts/benchmark_subpixel.py`](scripts/benchmark_subpixel.py)):

* **Total Trials**: 240
* **Tested Directions**: 8 directions (`+X`, `-X`, `+Y`, `-Y`, `diag_+X_+Y`, `diag_+X_-Y`, `diag_-X_+Y`, `diag_-X_-Y`)
* **Tested Displacement Magnitudes**: 6 magnitudes ($0.05, 0.10, 0.20, 0.30, 0.50, 0.75$ px)
* **Mean Absolute Error (MAE)**: $0.2416$ px
* **Median Absolute Error**: $0.2740$ px
* **Root Mean Square Error (RMSE)**: $0.2663$ px
* **95th Percentile Error (P95)**: $0.3844$ px
* **Maximum Error**: $0.3855$ px
* **Directional Bias (Bias X / Bias Y)**: $-0.0011$ px / $0.0000$ px (symmetric zero-centered)
* **Fraction < 0.10 px**: $13.3\%$
* **Fraction < 0.20 px**: $31.2\%$
* **Fraction < 0.25 px**: $40.4\%$
* **Fraction < 0.50 px**: $100.0\%$
* **Fraction < 1.00 px**: $100.0\%$
* *Empirical Finding: On the tested synthetic lunar-terrain benchmark, the Fourier phase-correlation refinement achieved approximately 0.24 px MAE, with all tested trials below 0.5 px. Accuracy is displacement-dependent and the benchmark does not establish universal <0.2 px precision. Peak refinement uses local log-domain peak interpolation intended to reduce interpolation bias.*

### B. Multi-Sensor Dataset Benchmark & 3-Way Triplet Cycle Consistency
Evaluated across all demonstration Chandrayaan-2 lunar regions and triplets ([`scripts/benchmark_registration.py`](scripts/benchmark_registration.py)):

#### 1. Multi-Sensor Bidirectional Pairwise & Triplet Cycle Results
| Dataset ID | OHRC ↔ TMC-2 | OHRC ↔ IIRS (Fwd / Rev) | TMC-2 ↔ IIRS (Fwd / Rev) | 3-Way Cycle Status ($A \to B \to C \to A$) | Cycle RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `region_001` | 7 inliers (1.85 px) | Rej (6 inl) / Rej (4 inl) | Rej (7 inl) / Insuff (0 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `region_002` | 7 inliers (1.24 px) | Rej (6 inl) / Rej (2 inl) | Rej (7 inl) / Insuff (0 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `region_003` | 6 inliers (1.77 px) | Rej (6 inl) / Rej (2 inl) | Rej (6 inl) / Insuff (0 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `region_004` | 7 inliers (1.78 px) | Rej (5 inl) / Insuff (0 inl) | Rej (7 inl) / Insuff (0 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `region_005` | 6 inliers (1.40 px) | Rej (5 inl) / Rej (4 inl) | Rej (7 inl) / Rej (5 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `region_006` | 6 inliers (1.65 px) | Rej (5 inl) / **5 inl (1.43 px)** | Rej (5 inl) / Rej (4 inl) | `cycle_not_computable` (BC failed) | *None* |
| `triplet_01_ch2_ohr_ncp_202` | 7 inliers (2.20 px) | **6 inl (1.34 px)** / Insuff (0 inl) | Rej (7 inl) / Rej (5 inl) | `cycle_not_computable` (BC, CA failed) | *None* |
| `triplet_new_2022` | 6 inliers (2.07 px) | **5 inl (0.92 px)** / Rej (4 inl) | **5 inl (0.26 px)** / Rej (5 inl) | `cycle_not_computable` (CA failed) | *None* |

*Note: In the table above, "Rej" indicates candidate matches failed geometric verification (ill-conditioned matrix / planar distortion), and "Insuff" indicates fewer than 4 verified correspondences.*

#### 2. Technical Audit: Why Direction Matters in Cross-Scale Matching
The registration pipeline exhibits directional sensitivity when pairing sensors across extreme scale disparity (~300× between OHRC and IIRS, ~13× between TMC-2 and IIRS):
1. **Asymmetric Grid Sampling**: Feature template centers are placed on a uniform $10 \times 10$ grid over `Image 1`. When `Image 1` is OHRC, downsampling to working scale ($69\text{ m/px}$) reduces the $512 \times 512$ tile to $40 \times 40$ pixels, dropping boundary cells and restricting candidates to the central tile. When `Image 1` is IIRS, templates are sampled across the full $512 \times 512$ canvas.
2. **Search Window Dynamics**: Template matching searches within `Image 2`. Searching a $16 \times 16$ template inside a $512 \times 512$ image (Image 2 = IIRS) tests a wide $170 \times 170$ search region, whereas searching inside a $40 \times 40$ image (Image 2 = OHRC) constrains the search region to $32 \times 32$ pixels.
3. **Conditioning of $H$ vs $H^{-1}$**: Mapping from a small spatial distribution to a wide distribution vs. the reverse creates distinct singular value ratios and projectivity coefficients in `verify_transformation_quality()`. For example, on `region_006`, `IIRS -> OHRC` satisfies geometric quality gates (5 inliers, 1.43 px RMSE), while `OHRC -> IIRS` is rejected due to collinearity.

#### 3. Strict Enforcement of the "Zero Synthetic Fallback" Principle
In earlier iterations, when an intermediate leg of the 3-way circular loop ($A \to B \to C \to A$) failed, an identity matrix ($I_{3\times 3}$) was substituted into the chain. This produced physically meaningless "cycle RMSE" values (such as 2685.5 px on `region_006`). 
- In [`data_preprocessing_pipeline/triplet_evaluator.py`](data_preprocessing_pipeline/triplet_evaluator.py), identity-matrix fallbacks are completely eliminated.
- If ANY of the three legs ($A \to B$, $B \to C$, or $C \to A$) fails geometric verification, the cycle error is not computed. The system returns `status: "cycle_not_computable"` with a `reason` naming the failed leg(s) and `cycle_rmse_px: null`.
- Only when all three physical legs genuinely produce verified transformations will a real numeric cycle RMSE be reported. Across the 8 current demonstration crops, all 8 have at least one unclosed leg in direct circular matching, correctly reporting `cycle_not_computable`.

---

## 7. SIH Problem Statement 26166 Delivery Matrix

| Requirement from Problem Statement | Status | Technical Evidence in Repository |
| :--- | :--- | :--- |
| **OHRC ↔ TMC-2 Cross-Registration** | Delivered (Primary) | Primary CFOG/Phase Congruency engine in [`ML_model/matcher_cfog.py`](ML_model/matcher_cfog.py) |
| **Multi-Modal Hyperspectral (IIRS)** | Delivered (Co-Registration) | Multi-band hyperspectral IIRS ingestion and physical-GSD-aware co-registration verified through a known-ground-truth synthetic IIRS integration test in [`tests/test_registration_pipeline.py`](tests/test_registration_pipeline.py). Due to IIRS's coarse spatial resolution (~75 m), this validates spatial co-registration rather than sub-meter feature correspondence. |
| **Scale Disparity Handling** | Delivered | Common physical-GSD normalization in [`ML_model/matcher_cfog.py`](ML_model/matcher_cfog.py); verified via known-ground-truth synthetic 20× physical-scale integration test in [`tests/test_registration_pipeline.py`](tests/test_registration_pipeline.py) |
| **Sun-Angle / Illumination Robustness** | Delivered | 2D Log-Gabor Phase Congruency & CFOG frequency structural features |
| **Spatially Distributed Matches** | Delivered | Grid binning & spatial distribution metrics in [`ML_model/metrics.py`](ML_model/metrics.py) |
| **Sub-Pixel Refinement** | Delivered | 2D Fourier Phase Correlation benchmarked at ~0.24 px MAE across 240 trials |
| **Held-Out Inlier Correspondence Validation** | Delivered | In-sample Fit RMSE separated from Held-Out Inlier Validation RMSE in [`ML_model/metrics.py`](ML_model/metrics.py) |
| **Terrain Parallax Compensation** | Delivered | Local DEM-based relief displacement compensation in [`ML_model/matcher_cfog.py`](ML_model/matcher_cfog.py) |
| **Full Output Product Package** | Delivered | Registered GeoTIFF (`.tif`), preview (`.png`), checkerboard (`.png`), JSON sidecars |
| **Zero Fake Fallbacks** | Verified | Clean error states on failure; zero manufactured corner points |

---

## 8. Installation & Usage Guide

### Prerequisites
* Python 3.10+ (macOS, Linux, Windows WSL)
* Node.js 18+ (for Next.js frontend)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Fable98/chandrayaan2-crossmatch.git
cd chandrayaan2-crossmatch

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Install frontend dependencies
cd lunar-frontend && npm install && cd ..
```

### Running the Deterministic Demo
```bash
python3 scripts/demo_registration.py \
  --source data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png \
  --reference data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png \
  --dem data_preprocessing_pipeline/processed_triplets/region_001/dem_512.png \
  --output registration_output_demo
```

### Running the Benchmarks
```bash
# Synthetic ground-truth subpixel benchmark
python3 scripts/benchmark_subpixel.py

# Multi-region registration benchmark
python3 scripts/benchmark_registration.py
```

### Running the Test Suite
```bash
pytest backend/test_api.py tests/test_registration_pipeline.py -v
```

### Starting the Production Servers Locally
```bash
# Terminal 1: FastAPI Backend (Port 8000)
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Next.js Mission Console (Port 3000)
cd lunar-frontend && npm run dev
```

---

## 9. Limitations & Future Scope

1. **Planar Projective Approximation**: The homography model operates as a local projective approximation. On steep lunar crater walls (>30° slope), non-planar relief displacement can induce localized residual errors.
2. **DEM Relief Compensation**: Relief displacement compensation currently uses local vertical height offsets rather than full iterative photogrammetric ray-intersection with a lunar orbital sensor model.
3. **IIRS Resolution Boundary**: IIRS GSD (~70–80 m) limits direct optical tie-point extraction. Hyperspectral information is integrated through co-registration rather than sub-meter feature correspondence.

---

## 10. Authoritative References

1. **ISRO Chandrayaan-2 Payload Documentation**: ISSDC/PRADAN Planetary Data System (PDS4) standards for OHRC, TMC-2, and IIRS.
2. **Phase Congruency**: Kovesi, P. (2000). *Phase Congruency Detects Corners and Edges*. The Australian Pattern Recognition Society Conference (DICTA 2000).
3. **CFOG Descriptor**: Ye, Y., Shan, J., Hao, S., Bruzzone, L., & Qin, Y. (2019). *A Local Feature Descriptor Based on Channel Features of Oriented Gradients for Multispectral Remote Sensing Image Registration*. IEEE Transactions on Geoscience and Remote Sensing (TGRS), 58(4), 2310-2321.
4. **LoFTR Baseline**: Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). *LoFTR: Detector-Free Local Feature Matching with Transformers*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
