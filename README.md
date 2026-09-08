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
- **Spatial Uniformity Score:** Measures how evenly distributed the match points are across the image grid.
- **Inlier Ratio:** Percentage of raw matches that pass rigorous geometric verification.

All metrics are computed via a single canonical module (`ML_model/metrics.py`) ensuring consistency between the live dashboard, batch benchmarks, and the official ISRO evaluator.

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
4. **LoFTR Baseline:** Sun, J. et al. (2021). *LoFTR: Detector-Free Local Feature Matching with Transformers*. IEEE/CVF CVPR.
