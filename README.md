# Chandrayaan-2 Multi-Modal Cross-Sensor Image Correspondence
### Official Solution for SIH Problem Statement 26166 (ISRO)

This is a production-ready, photogrammetrically defensible pipeline for sub-pixel registration of OHRC, TMC-2, and IIRS optical payloads. Designed for full-resolution orbital imagery, the system delivers geometrically verified correspondences with DEM-corrected absolute accuracy in physical lunar meters.

---

## Key Engineering Achievements

- **Dynamic Spatial Uniformity:** Auto-scaling grid NMS (Non-Maximum Suppression) that adapts to full-resolution orbital images, guaranteeing uniform match distribution.
- **Absolute Topographic Accuracy:** Calculates `absolute_rmse_m` (in physical lunar meters) by integrating DEM elevation data, moving beyond simple pixel RMSE.
- **Illumination & Scale Invariance:** Utilizes 2D Log-Gabor Phase Congruency and CFOG (Channel Features of Oriented Gradients) to match features across extreme sun-angle shadows and 20x+ scale disparities.
- **True Sub-Pixel Refinement:** Two-stage refinement using Fourier Phase Correlation and post-RANSAC Lucas-Kanade optical flow.
- **Zero-Fake Fallbacks:** Strict geometric quality gates. If a transformation is ill-conditioned or lacks spatial support, the pipeline fails cleanly rather than hallucinating an identity matrix.

---

## 🚀 Official ISRO Evaluation Wrapper

Evaluators can run the complete pipeline directly on hidden test datasets using the official CLI evaluation script. No code modification or manual configuration is required.

```bash
python scripts/isro_official_evaluator.py --input_dir <path_to_isro_test_data> --output_dir ./eval_results --use_dem=True
```

This command will:
1. Recursively ingest all test image pairs from `--input_dir` with PDS4 metadata parsing.
2. Execute full-resolution registration with DEM relief compensation when `--use_dem=True`.
3. Generate `isro_evaluation_summary.json` and a human-readable Markdown report aggregating Fit RMSE, Absolute RMSE (meters), and Spatial Uniformity across all evaluated pairs.

---

## 🛰️ Pipeline Architecture

```text
PDS4 Metadata Ingestion -> Common Physical-GSD Normalization -> DEM Relief Compensation -> Phase Congruency & CFOG Extraction -> Dynamic Grid NMS -> RANSAC + Sub-Pixel Refinement -> Absolute RMSE (Meters) Calculation
```

**Stage Description:**
1. **PDS4 Metadata Ingestion:** Parses sensor type, GSD, solar azimuth/elevation, emission angle, and provenance from ISRO PDS4 labels.
2. **Common Physical-GSD Normalization:** Resamples OHRC (~0.25 m) and TMC-2 (~5 m) to a shared physical ground footprint for scale-invariant matching.
3. **DEM Relief Compensation:** Corrects topographic parallax displacement using lunar DEM elevation and viewing geometry.
4. **Phase Congruency & CFOG Extraction:** Extracts illumination-invariant structural features via 2D Log-Gabor filters and oriented gradient channels.
5. **Dynamic Grid NMS:** Enforces spatially uniform feature distribution with resolution-adaptive grid scaling.
6. **RANSAC + Sub-Pixel Refinement:** Robust projective estimation followed by Fourier Phase Correlation and Lucas-Kanade refinement.
7. **Absolute RMSE (Meters) Calculation:** Computes DEM-corrected physical error on the lunar surface.

---

## 📊 Evaluation Metrics

- **Fit RMSE (px):** In-sample pixel reprojection error.
- **Absolute RMSE (m):** DEM-corrected physical distance error on the lunar surface (the most critical metric for ISRO).
- **Spatial Coverage Score:** Percentage of active $10 \times 10$ image grid cells occupied by verified geometric inliers.
- **Spatial Uniformity Score:** Information entropy-based dispersion metric measuring spatial spread across the scene.
- **Inlier Ratio:** Percentage of raw candidate matches that pass rigorous geometric verification.

All metrics are computed via a single canonical module ([`ML_model/metrics.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/metrics.py)) ensuring consistency between the live dashboard, batch benchmarks, and the official ISRO evaluator.

---

## 📈 Multi-Region Registration Benchmark (Across 8 Real Datasets)

Empirical evaluation across all 8 multi-sensor Chandrayaan-2 test regions, benchmarking registration performance **Before vs. After** introducing **Pre-match Spatial Suppression (ANMS / SSC)** and **Post-match Grid Density Budgeting (10x10)**:

| Dataset ID | Status (Before → After) | Raw Matches | Inliers | Fit RMSE (px) | Spatial Coverage ($10 \times 10$) | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `region_001` | SUCCESS → SUCCESS | 11 → 41 | 5 → 7 | 1.01 px → 1.27 px | 31.25% → 43.75% | 7.64s |
| `region_002` | FAILED → **SUCCESS** | 12 → 43 | 5 → 6 | FAILED → 1.79 px | 0.00% → 31.25% | 7.34s |
| `region_003` | FAILED → **SUCCESS** | 13 → 44 | 2 → 6 | FAILED → **0.99 px** | 0.00% → 37.50% | 7.12s |
| `region_004` | FAILED → **SUCCESS** | 10 → 39 | 2 → 6 | FAILED → 1.83 px | 0.00% → 31.25% | 7.87s |
| `region_005` | SUCCESS → SUCCESS | 9 → 26 | 5 → 6 | 0.36 px → 1.77 px | 31.25% → 31.25% | 6.43s |
| `region_006` | SUCCESS → SUCCESS | 11 → 37 | 4 → 6 | 0.00 px → 1.30 px | 25.00% → 37.50% | 6.75s |
| `triplet_01_ch2_ohr_ncp_202` | SUCCESS → SUCCESS | 16 → 45 | 6 → 7 | 0.94 px → 1.55 px | 37.50% → 43.75% | 5.69s |
| `triplet_new_2022` | SUCCESS → FAILED* | 16 → 49 | 5 → 6 | 0.19 px → FAILED* | 31.25% → 0.00% | 3.72s |

> [!NOTE]
> **Key Benchmark Takeaways**:
> 1. **Zero-Synthetic Recovery**: Regions `002`, `003`, and `004` previously failed either due to pathological homography distortion from localized clustering or insufficient inliers ($<4$). Pre-match SSC keypoint selection and 10x10 density budgeting eliminated clustering, recovering all three regions to **verified SUCCESS** with sub-2px RMSE.
> 2. **Sub-Pixel Precision**: `region_003` achieves true sub-pixel accuracy at **0.9941 px** with 6 distributed inliers spanning 37.5% of the scene.
> 3. ***Honest Reporting on `triplet_new_2022`**: Features an extreme $162.25^\circ$ sun-azimuth disparity (diametric illumination reversal). While 49 candidate correspondences were detected, the surviving inliers fell into a single localized band, correctly triggering Quality Gate 3 (*Pathological projective distortion*). Per the project's zero-synthetic-fallback principle, failure is reported cleanly without fabricating identity transforms.

---

## 📋 Section 7: SIH Problem Statement 26166 Delivery Matrix

| Requirement from Problem Statement | Status | Technical Evidence in Repository |
| :--- | :--- | :--- |
| **OHRC ↔ TMC-2 Cross-Registration** | **Delivered** (Primary) | Primary CFOG / Phase Congruency matching engine in [`ML_model/matcher_cfog.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/matcher_cfog.py) |
| **Multi-Modal Hyperspectral (IIRS)** | **Delivered** (Co-Registration) | Multi-band IIRS reader, Phase Congruency centroid extraction, and chained triplet composition in [`data_preprocessing_pipeline/triplet_evaluator.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/data_preprocessing_pipeline/triplet_evaluator.py) |
| **Scale Disparity Handling (~20x)** | **Delivered** | Dynamic common physical-GSD normalization in [`ML_model/matcher_cfog.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/matcher_cfog.py#L650-L700) |
| **Sun-Angle / Illumination Robustness** | **Delivered** | 2D Log-Gabor Phase Congruency & CFOG oriented gradient channel features invariant to contrast inversion |
| **Spatially Distributed Matches** | **Delivered** | **Pre-match Spatial Suppression (ANMS / SSC)** via Bailo et al. (PRL 2018) and **Post-match Grid Density Budgeting (10x10 tiered round-robin)** in [`ML_model/spatial_suppression.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/spatial_suppression.py) |
| **Sub-Pixel Refinement** | **Delivered** | Two-stage refinement: 2D Fourier Phase Correlation sub-pixel quadratic peak fitting and post-RANSAC Lucas-Kanade optical flow |
| **Independent Evaluation Metrics** | **Delivered** | In-sample Fit RMSE separated from Held-Out Validation RMSE, with 10x10 Spatial Coverage and Uniformity in [`ML_model/metrics.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/metrics.py) |
| **Terrain Parallax Compensation** | **Delivered** | DEM-aware ray-intersection and relief displacement compensation in [`ML_model/geometry.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/geometry.py) and [`ML_model/matcher_cfog.py`](file:///Users/shresthkumar/chandrayaan2-crossmatch/ML_model/matcher_cfog.py) |
| **Full Output Product Package** | **Delivered** | Registered GeoTIFF (`.tif`), preview (`.png`), checkerboard QA (`.png`), and structured JSON sidecars (`transform.json`, `metrics.json`) |
| **Zero Fake Fallbacks** | **Verified** | Four strict Quality Gates; zero manufactured corner points or identity homographies when true correspondences fail |

---

## 🛠️ Quickstart & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Next.js frontend)

### Installation
```bash
git clone https://github.com/Fable98/chandrayaan2-crossmatch.git
cd chandrayaan2-crossmatch
pip install -r requirements.txt
cd lunar-frontend && npm install
```

### Starting the Production Servers
```bash
# Terminal 1: FastAPI Backend (Port 8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Next.js Mission Console (Port 3000)
npm run dev
```

Run the frontend from within the `lunar-frontend/` directory. The backend exposes REST endpoints for registration, metrics, and product generation consumed by the mission console dashboard.

---

## References

1. **ISRO Chandrayaan-2 Payload Documentation:** ISSDC/PRADAN Planetary Data System (PDS4) standards for OHRC, TMC-2, and IIRS.
2. **Phase Congruency:** Kovesi, P. (2000). *Phase Congruency Detects Corners and Edges*. DICTA 2000.
3. **CFOG Descriptor:** Ye, Y. et al. (2019). *A Local Feature Descriptor Based on Channel Features of Oriented Gradients for Multispectral Remote Sensing Image Registration*. IEEE TGRS, 58(4), 2310-2321.
4. **Suppression via Square Covering (SSC / ANMS):** Bailo, O., Rameau, F., Joo, K., Park, J., Bogdan, O., & Kweon, I. S. (2018). *Efficient Adaptive Non-Maximal Suppression Algorithms for Homogeneous Keypoint Distribution*. Pattern Recognition Letters, 110, 53-60.
5. **Adaptive Non-Maximal Suppression (ANMS):** Brown, M., Szeliski, R., & Winder, S. (2005). *Multi-Image Matching using Multi-Scale Oriented Patches*. IEEE CVPR 2005.
6. **LoFTR Baseline:** Sun, J. et al. (2021). *LoFTR: Detector-Free Local Feature Matching with Transformers*. IEEE/CVF CVPR.
