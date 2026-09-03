# Chandrayaan-2 Multi-Sensor Cross-Registration Engine (SIH26166)

Automated sub-pixel tie-point correspondence and projective co-registration software designed for Chandrayaan-2's Orbiter High-Resolution Camera (OHRC), Terrain Mapping Camera-2 (TMC-2), and Imaging Infrared Spectrometer (IIRS) instruments.

Developed for the Smart India Hackathon (SIH26166) problem statement proposed by the Indian Space Research Organisation (ISRO).

---

## 1. Problem Statement Overview

Planetary image registration across distinct sensors on the Chandrayaan-2 orbiter presents acute computer vision and photogrammetric challenges:

* **Orbiter High-Resolution Camera (OHRC)**: Panchromatic optical imagery at 0.25–0.32 m/pixel ground sampling distance (GSD).
* **Terrain Mapping Camera-2 (TMC-2)**: Stereo optical imagery at ~4–5 m/pixel GSD (Fore, Aft, and Nadir views).
* **Imaging Infrared Spectrometer (IIRS)**: Hyperspectral imagery at ~70–80 m/pixel GSD spanning 256 contiguous spectral channels (0.8–5.0 um).

### Core Photogrammetric and Geometric Challenges

1. **Extreme Scale Disparity**: An 18x–20x spatial resolution gap exists between OHRC and TMC-2, and an ~280x gap between OHRC and IIRS. Traditional patch-based cross-correlation fails because high-frequency textural features visible in OHRC collapse into single pixels in TMC-2 and sub-pixel fractions in IIRS.
2. **Drastic Illumination Inversions**: Non-repeat polar orbits produce image acquisitions under wildly divergent solar azimuth and elevation angles (>160 degrees difference). Craters illuminate from opposite sides, producing inverted shadows that cause intensity-based metrics (such as MSE, SSIM, and normalized cross-correlation) to converge on false local minima.
3. **Sub-Pixel Accuracy Demands**: Scientific analysis, DEM generation, and precision landing hazard avoidance require tie-point registration accuracy below 1.0 pixel RMSE.
4. **Spatial Uniformity**: Feature detectors typically cluster hundreds of keypoints along high-contrast crater rims while completely ignoring flat lunar maria, producing degenerate geometric transformations when computing planar homographies or polynomial warps.

---

## 2. Requirement Comparison: Demanded vs. Delivered

The table below directly benchmarks the ISRO SIH26166 problem statement requirements against the software capabilities implemented in this repository:

| Requirement (ISRO SIH26166) | Demanded Specification | Implementation Status | Implementation Details |
| :--- | :--- | :---: | :--- |
| **Generic Software Solution** | Must process arbitrary, previously unseen image pairs dynamically without hardcoded coordinates. | **Delivered** | Dynamic `POST /register` endpoint accepts arbitrary image uploads (GeoTIFF, PNG, JPEG), executes on-the-fly feature matching, computes homography, and returns registered products. |
| **Sub-Pixel Accuracy** | Precision better than 1.0 pixel (e.g., < 0.5 pixel mean error). | **Delivered** | Two-stage matching: LoFTR transformer provides coarse correspondences, followed by 2D Fourier Phase Correlation with sub-pixel quadratic peak interpolation on extracted patches. |
| **Spatial Uniformity** | Matches distributed across the full image domain, not clustered on crater rims. | **Delivered** | 10x10 spatial non-maximal suppression grid. The image is divided into 100 cells, capping matches to the top 5 highest-confidence points per cell to enforce broad spatial coverage. |
| **Aspect-Ratio Preservation** | Scaled coordinates must match unscaled sensor geometry without aspect-ratio distortion. | **Delivered** | Dynamic resizing maps coordinates back to original unscaled pixel space using explicit independent coordinate scale factors before computing homography. |
| **Registered Output Product** | Software must produce a warped product and visual proof of alignment. | **Delivered** | Generates perspective-warped source images (`warped_source.jpg`) and interactive 50px alternating checkerboard composite products (`registered_checkerboard.jpg`). |
| **Scientific Metric Evaluation** | Quantitative accuracy metrics must be calculated and exposed. | **Delivered** | Calculates Root Mean Square Error (RMSE), total inlier count, true RANSAC inlier ratio, uniformity distribution score, and sub-pixel pass/fail validation. |
| **Multi-Modal Hyperspectral Ingestion** | Ingestion of multi-band hyperspectral cubes (>3 channels). | **Delivered** | Rasterio and tifffile pipeline detects cubes with >3 channels and computes a calibrated pseudo-panchromatic intensity map via spectral averaging across valid bands. |
| **Illumination Robustness** | Must handle severe solar incidence variations without divergence. | **Delivered** | Transformer-based dense local feature matching (LoFTR) learns global contextual relationships rather than relying on local intensity gradients. |

---

## 3. Engineering Architecture & Pipeline

```
+-----------------------------------------------------------------------------------+
|                            Input Multi-Sensor Imagery                             |
|          OHRC (0.25 m/px)   |   TMC-2 (~4-5 m/px)   |   IIRS (~70-80 m/px)        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 1: Robust Sensor Ingestion & Preprocessing (`ML_model/matcher.py`)          |
| - Format-agnostic loader: GeoTIFF, TIFF, PNG, JPEG via Rasterio & OpenCV          |
| - Hyperspectral handling: Multi-band cubes (>3 channels) converted to             |
|   1-channel pseudo-panchromatic intensity maps via spectral mean reduction        |
| - Bit-depth normalization: Standardized to uint8 dynamic range                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 2: Scale-Invariant Deep Feature Matching (LoFTR)                            |
| - Internal 512x512 inference resolution with scale tracking                       |
| - Linear Transformer self- and cross-attention across sensor representations       |
| - Coordinate projection back to original pixel dimensions                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3: Spatial Uniformity Grid Filter                                            |
| - 10x10 spatial grid partitioning over source image dimensions                    |
| - Confidence-based non-maximal suppression (top 5 points per grid cell)           |
| - Prevents crater rim clustering and enforces distributed coverage                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 4: Sub-Pixel Refinement via Phase Correlation                               |
| - Local patch extraction (32x32 window) around candidate tie-points                |
| - 2D Fourier Cross-Power Spectrum computation                                     |
| - Quadratic surface fitting for sub-pixel peak interpolation (<0.2 px precision)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 5: RANSAC Homography & Registration Product Generation                      |
| - Robust planar homography matrix estimation (RANSAC with 3.0 px threshold)       |
| - Projective transformation of source image (`cv2.warpPerspective`)               |
| - 50px alternating checkerboard composite blend for visual verification           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 6: Telemetry & Error Metrics                                                |
| - RMSE (Root Mean Square Error in pixels)                                         |
| - Inlier count & true RANSAC inlier ratio                                         |
| - Spatial uniformity coverage percentage                                          |
| - Sub-pixel verification flag (RMSE < 1.0 px)                                     |
+-----------------------------------------------------------------------------------+
```

---

## 4. Scope & Scientific Demarcation

* **Primary Registration Engine (OHRC <-> TMC-2)**: High-resolution panchromatic optical data (OHRC) is registered to stereo context optical data (TMC-2). The deep matching network and phase correlation algorithms operate directly between these two sensors.
* **IIRS Hyperspectral Status**: IIRS data has a native spatial resolution of ~70–80 m/pixel across 256 spectral channels. In this software:
  * Multi-band GeoTIFF cubes are ingested and converted to calibrated intensity layers.
  * IIRS mineralogy layers are presented as co-registered spatial overlays within the GIS viewer.
  * Direct 256-band hyperspectral tie-point matching is classified as an experimental stretch capability due to physical resolution limits (80 m information cannot synthetically yield 0.25 m spatial features).
* **Terminology**: This system uses the scientifically verified designation **illumination-robust**, reflecting deep cross-attention invariance to solar angles rather than unphysical claims of absolute illumination invariance.

---

## 5. Repository Layout

```
chandrayaan2-crossmatch/
├── ML_model/
│   ├── matcher.py              # Core LoFTR, 10x10 grid filter, phase correlation & warping
│   └── loftr_outdoor.ckpt      # Pre-trained deep matching weights
├── backend/
│   ├── main.py                 # FastAPI application & dynamic /register endpoint
│   ├── config.py               # Pydantic environment configuration
│   ├── schemas.py              # Request/response validation schemas
│   ├── test_api.py             # Pytest automated test suite (34 test cases)
│   ├── requirements.txt        # Backend dependencies (FastAPI, Uvicorn, OpenCV, Rasterio)
│   ├── data/
│   │   └── loader.py           # In-memory triplet catalog and match point indexing
│   └── routers/
│       ├── triplets.py         # Triplet catalog endpoints
│       ├── matches.py          # Tie-point and telemetry endpoints
│       └── images.py           # Tile, DEM, and IIRS imagery streaming
├── lunar-frontend/             # Production Next.js 14 Web Application
│   ├── src/
│   │   ├── app/                # Next.js App Router root layout and page
│   │   ├── components/
│   │   │   ├── Console.tsx             # Primary mission registration console
│   │   │   ├── MapPanel.tsx            # Leaflet-based multi-payload lunar GIS map
│   │   │   ├── LinkedCursorPanel.tsx   # Interactive side-by-side tie-point inspection
│   │   │   ├── RegistrationLauncher.tsx# Dynamic file upload & live registration modal
│   │   │   ├── archive/                # Dossier, Vault, and Theory technical modals
│   │   │   └── hero/                   # 3D interactive lunar globe landing interface
│   │   └── lib/                # API client, coordinate math, and TypeScript interfaces
│   ├── package.json
│   ├── vercel.json             # Vercel deployment configuration
│   └── tailwind.config.js
├── data_preprocessing_pipeline/
│   ├── processed_triplets/     # Pre-processed lunar region datasets
│   └── config/default.yaml     # Ingestion GSD and projection profiles
├── requirements.txt            # Root Python dependencies
└── render.yaml                 # Render infrastructure-as-code deployment specification
```

---

## 6. Installation and Setup

### Prerequisites

* Python 3.10 to 3.12 (Python 3.11 recommended)
* Node.js 18.x or later (with npm 9+)
* Git LFS (if cloning model checkpoints)

### 1. Clone the Repository

```bash
git clone https://github.com/Fable98/chandrayaan2-crossmatch.git
cd chandrayaan2-crossmatch
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the test suite to verify installation
pytest backend/test_api.py
```

### 3. Frontend Setup

```bash
cd lunar-frontend
npm install
cp .env.local.example .env.local
cd ..
```

---

## 7. Running the Project Locally

To run the complete system, start the backend and frontend in separate terminal windows:

### Terminal 1: FastAPI Backend Service

```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* API Base URL: `http://localhost:8000`
* Health Check: `http://localhost:8000/health`
* Interactive API Documentation (Swagger): `http://localhost:8000/docs`

### Terminal 2: Next.js Frontend

```bash
cd lunar-frontend
npm run dev
```

* Web Application: `http://localhost:3000`
* Direct Mission Console: `http://localhost:3000/?view=console`

---

## 8. Deployment Architecture

| Component | Platform | URL | Configuration |
| :--- | :--- | :--- | :--- |
| **Frontend** | Vercel | `https://chandrayaan2-crossmatch-sand.vercel.app` | Next.js 14 App Router, auto-deploy from `main` branch with root directory `lunar-frontend`. |
| **Backend** | Render | `https://chandrayaan2-crossmatch.onrender.com` | FastAPI with Uvicorn, Python 3.11 environment. |

### Memory Optimization for Free Tier Hosting

To operate reliably within constrained cloud environments (such as Render's 512 MB RAM free tier):
* **Lazy Loading**: Heavy dependencies (`torch`, `kornia`, `LoFTR`) are imported on-demand inside `match_images` rather than during module initialization. The API boots in under 20 MB of RAM.
* **Explicit Garbage Collection**: `gc.collect()` and `torch.cuda.empty_cache()` are invoked immediately after registration inference to release memory buffers.
* **CORS Security**: `backend/main.py` uses regex origin validation (`allow_origin_regex=r"https://.*\.vercel\.app"`) to support preview and production frontend domains without manual origin whitelisting.

---

## 9. Verification & Testing

### Backend Unit Tests

The backend test suite verifies schema conformance, catalog retrieval, tile streaming, error handling, and dynamic registration validation:

```bash
pytest backend/test_api.py -v
```

All 34 tests execute and pass in under one second.

### Frontend Compilation

Verify TypeScript types and production bundle generation:

```bash
cd lunar-frontend
npm run build
```

The Next.js build compiles all routes cleanly with zero linting or type errors.

---

## 10. License

Developed under the Smart India Hackathon (SIH 2024 / SIH26166) initiative for research and academic evaluation under ISRO problem statement specifications.
