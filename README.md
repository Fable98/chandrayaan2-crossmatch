# Chandrayaan-2 Cross-Sensor Image Correspondence

### Sun-angle and scale-aware image correspondence across OHRC, TMC-2 and IIRS

> **Smart India Hackathon — SIH26166 · ISRO**

A cross-sensor lunar image correspondence and registration system for Chandrayaan-2 imagery. The project establishes spatial correspondence between the **Orbiter High Resolution Camera (OHRC)**, **Terrain Mapping Camera-2 (TMC-2)** and **Imaging Infrared Spectrometer (IIRS)** by combining metadata-driven georeferencing, common-scale normalization, illumination-aware representations, deep feature matching, geometric verification and visual registration QA.

The repository also includes a FastAPI backend and an interactive Next.js frontend for inspecting lunar regions, match correspondences, geographic footprints, registration products and evaluation metrics.

---

## Table of Contents

* [Overview](#overview)
* [Problem](#problem)
* [What the System Does](#what-the-system-does)
* [Sensors](#sensors)
* [System Architecture](#system-architecture)
* [End-to-End Pipeline](#end-to-end-pipeline)
* [1. Data Ingestion](#1-data-ingestion)
* [2. Sensor Preprocessing](#2-sensor-preprocessing)
* [3. Lunar Georeferencing](#3-lunar-georeferencing)
* [4. Illumination Normalization](#4-illumination-normalization)
* [5. Scale Normalization](#5-scale-normalization)
* [6. Multi-Representation Matching](#6-multi-representation-matching)
* [7. LoFTR Feature Matching](#7-loftr-feature-matching)
* [8. Geometric Verification](#8-geometric-verification)
* [9. Spatial Uniformity Filtering](#9-spatial-uniformity-filtering)
* [10. Registration](#10-registration)
* [11. Evaluation](#11-evaluation)
* [12. Backend](#12-backend)
* [13. Frontend](#13-frontend)
* [Repository Structure](#repository-structure)
* [Data Products](#data-products)
* [Configuration](#configuration)
* [Installation](#installation)
* [Running the Preprocessing Pipeline](#running-the-preprocessing-pipeline)
* [Running Feature Matching](#running-feature-matching)
* [Generating Registration Products](#generating-registration-products)
* [Running the Backend](#running-the-backend)
* [Running the Frontend](#running-the-frontend)
* [Live Registration](#live-registration)
* [API](#api)
* [Evaluation Metrics](#evaluation-metrics)
* [Example Region](#example-region)
* [Important Implementation Notes](#important-implementation-notes)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [References](#references)
* [License](#license)

---

# Overview

Chandrayaan-2 acquired lunar surface imagery using multiple scientific payloads with substantially different spatial resolutions, acquisition geometries, spectral characteristics and illumination conditions.

This creates a difficult image correspondence problem:

```text
                Chandrayaan-2 Lunar Data
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        OHRC            TMC-2          IIRS
       ~0.25 m          ~5 m           ~69 m
          │              │              │
          └──────────────┼──────────────┘
                         │
                 Common Geospatial
                    Representation
                         │
              Illumination / Scale
                     Normalization
                         │
             Multi-Representation Image
                       Matching
                         │
                       LoFTR
                         │
                      RANSAC
                         │
                Spatial Filtering
                         │
                    Homography
                         │
                Registered Products
                         │
              Quantitative Evaluation
```

The central design principle is to avoid asking a feature matcher to solve every source of variation simultaneously.

Instead, the system progressively reduces:

* geographic misalignment,
* sensor-dependent scale differences,
* illumination/shadow differences,
* representation differences,
* sparse or spatially clustered correspondences.

---

# Problem

The same lunar terrain can look dramatically different between sensors.

For example:

* OHRC provides extremely high-resolution imagery.
* TMC-2 observes the same terrain at a much coarser ground sampling distance.
* IIRS provides substantially different spectral information.
* The sensors can observe the same region under different solar geometries.
* Shadows and terrain contrast can change significantly.
* Pixel coordinates are not directly comparable before spatial normalization.

A naive pixel-to-pixel comparison therefore fails.

The project treats the problem as:

> **Find reliable correspondences between images representing the same lunar terrain despite differences in spatial scale, illumination, sensor characteristics and image representation.**

---

# What the System Does

The project provides two related workflows.

## Offline scientific pipeline

The preprocessing pipeline converts raw or converted lunar products into a common representation:

```text
PDS4 XML + image data
        │
        ▼
Metadata extraction
        │
        ▼
Sensor-specific cleanup
        │
        ▼
Lunar georeferencing
        │
        ▼
Photometric normalization
        │
        ▼
Common working GSD
        │
        ▼
Invariant representations
        │
        ▼
512 × 512 tiles
        │
        ▼
OHRC / TMC-2 / IIRS triplets
```

## Correspondence and registration pipeline

```text
OHRC ───────────────┐
                    │
TMC-2 ──────────────┼──► LoFTR
                    │
IIRS ───────────────┘
                       │
                       ▼
                  Raw matches
                       │
                       ▼
                    RANSAC
                       │
                       ▼
              Spatial filtering
                       │
                       ▼
                  Homography
                       │
                       ▼
             Registered OHRC
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Blend overlay       Checkerboard QA
```

---

# Sensors

The repository works with three Chandrayaan-2 optical payloads.

| Sensor    | Typical GSD in project data | Role                                            |
| --------- | --------------------------: | ----------------------------------------------- |
| **OHRC**  |                     ~0.25 m | High-resolution reference/source imagery        |
| **TMC-2** |                        ~5 m | Medium-resolution terrain mapping imagery       |
| **IIRS**  |                    ~69–80 m | Coarser multispectral/hyperspectral information |

The actual GSD is read from product metadata where available rather than assuming one universal value.

For example, `region_001` records:

* OHRC: `0.25 m`
* TMC-2: `5.4 m`
* IIRS: `69.04 m`

along with sensor footprints and solar azimuth information.

---

# System Architecture

The repository is organized into several functional layers:

```text
chandrayaan2-crossmatch/
│
├── data_preprocessing_pipeline/
│   ├── lunar_pipeline/
│   ├── config/
│   ├── scripts/
│   ├── matches/
│   └── processed_triplets/
│
├── ML_model/
│   └── matcher.py
│
├── scripts/
│   ├── run_loftr_all_regions.py
│   ├── generate_enhanced_matches.py
│   ├── register.py
│   ├── distribution.py
│   └── check_iirs_rotation.py
│
├── lunar_project/
│   └── src/ml/evaluation/
│
├── backend/
│   ├── main.py
│   ├── geo.py
│   ├── data/
│   ├── routers/
│   └── schemas.py
│
├── lunar-frontend/
│   └── Next.js application
│
├── evaluation_output/
├── registration_output/
└── processed_user/
```

---

# End-to-End Pipeline

## Stage 1 — Ingest

Input products can include:

* PDS4 XML-labelled products
* `.img`
* `.IMG`
* `.dat`
* `.qub`
* GeoTIFF
* TIFF
* ISIS cubes
* JPEG2000 products

The ingestion layer detects PDS4 labels, resolves associated image files and extracts metadata.

Relevant implementation:

```text
data_preprocessing_pipeline/lunar_pipeline/ingest.py
```

The ingestion code extracts information such as:

* product identifier,
* sensor,
* acquisition time,
* footprint,
* longitude/latitude bounds,
* GSD,
* solar azimuth,
* solar elevation,
* incidence angle,
* emission angle,
* phase angle.

---

# 1. Data Ingestion

The ingestion system supports two primary cases.

### PDS4-labelled data

```text
product.xml
product.img
```

The XML label is parsed first. If the raster can be opened through Rasterio/GDAL, that representation is used.

If it cannot, the pipeline contains a raw binary fallback which uses the dimensions and data type described in the PDS4 label.

### Already-converted raster products

GeoTIFF and compatible raster formats can be loaded directly.

A matching XML sidecar is also used when available to supplement raster metadata.

---

# 2. Sensor Preprocessing

Sensor-specific cleanup is implemented in:

```text
data_preprocessing_pipeline/lunar_pipeline/sensors.py
```

## Pushbroom destriping

For TMC/OHRC-type imagery, the pipeline performs column-wise median normalization.

Conceptually:

```text
column median
      │
      ▼
compare with global median
      │
      ▼
derive correction factor
      │
      ▼
normalize column
```

This attempts to reduce systematic striping while preserving the underlying terrain structure.

## IIRS dimensionality reduction

IIRS can contain multiple bands.

The pipeline supports:

```yaml
iirs_reduce: pca
```

or:

```yaml
iirs_reduce: band
```

With PCA enabled, the spectral cube is projected into a smaller number of components before matching.

---

# 3. Lunar Georeferencing

The project uses a lunar equirectangular coordinate system based on the Moon's mean radius:

```text
R = 1,737,400 m
```

The default CRS is:

```text
+proj=eqc
+lat_ts=0
+lon_0=0
+a=1737400
+b=1737400
+units=m
```

The georeferencing module can:

1. use existing raster CRS/transform information,
2. assign georeferencing from product footprints,
3. reproject imagery into the common lunar CRS,
4. optionally attempt an ISIS3 import path.

---

# Shared Triplet Coordinate System

Once images are converted into a validated triplet, the project uses a shared bounding box:

```text
west_lon
east_lon
south_lat
north_lat
```

The common 512 × 512 representation then uses:

```text
lon = west_lon + (px / 512) × (east_lon - west_lon)

lat = north_lat - (py / 512) × (north_lat - south_lat)
```

This means corresponding pixels in the processed triplet have a direct geographic interpretation.

The backend intentionally uses this shared-bounds model for its pixel-to-lat/lon conversion.

The repository preserves an older corner-based perspective conversion implementation, but the live backend path uses the shared bounding-box affine mapping.

---

# 4. Illumination Normalization

Different acquisition times can produce substantially different lunar shadows.

The pipeline therefore includes photometric normalization.

Supported models:

```yaml
photometric_model: lunar_lambert
```

or:

```yaml
photometric_model: hapke
```

or:

```yaml
photometric_model: none
```

## Lunar-Lambert

The implementation combines Lambertian and Lommel-Seeliger terms:

```text
f = (1-L) × Lommel-Seeliger + L × Lambert
```

where the phase angle controls the mixture.

## Hapke

A simplified Hapke-inspired disk function is also implemented.

This is intentionally a lightweight normalization model rather than a complete physical photometric inversion.

---

# Shadow Detection

A shadow mask can be generated using a low-intensity percentile threshold.

Default:

```yaml
shadow_percentile: 3.0
```

If incidence angle exceeds:

```yaml
incidence_shadow_deg: 85.0
```

the implementation can mark the entire image as shadow-dominated.

The mask is useful for downstream quality control and matching diagnostics.

---

# 5. Scale Normalization

The sensors operate at dramatically different resolutions.

The project therefore defines a common working GSD:

```yaml
working_gsd_m: 5.0
```

This is a compromise between:

* OHRC's very fine resolution,
* TMC-2's medium resolution,
* IIRS's much coarser resolution.

For example, OHRC is downsampled toward the working scale instead of forcing LoFTR to simultaneously solve extreme scale differences.

The scale module also supports Gaussian pyramids for multi-resolution representations.

---

# 6. Multi-Representation Matching

Raw intensity is not the only representation used.

The preprocessing pipeline can generate:

### Census representation

A local binary comparison representation designed to reduce sensitivity to absolute intensity changes.

### Gradient orientation

The image gradient is represented using:

```text
cos(theta)
sin(theta)
```

rather than raw intensity.

This emphasizes structural edges such as:

* crater rims,
* ridges,
* valleys,
* terrain boundaries.

### Local Binary Pattern — LBP

LBP captures local texture structure.

### Phase-congruency proxy

A lightweight multi-scale local-energy representation is available in the implementation, although it is not part of the default invariant list.

The default configuration enables:

```yaml
invariant_modes:
  - census
  - gradient
  - lbp
```

---

# 7. LoFTR Feature Matching

The main feature correspondence model is **LoFTR — Detector-Free Local Feature Matching with Transformers**.

The repository uses Kornia's LoFTR implementation.

The standard matching flow is:

```text
Image A
   │
   ▼
512 × 512 normalization
   │
   ▼
LoFTR
   │
   ├── keypoints0
   ├── keypoints1
   └── confidence
          │
          ▼
       RANSAC
```

The primary multi-region implementation is:

```text
scripts/run_loftr_all_regions.py
```

It processes the available region directories and performs OHRC/TMC matching.

The script also supports matching invariant representations such as the census representation.

---

# 8. Geometric Verification

Raw neural matches may contain incorrect correspondences.

The project therefore estimates a homography using OpenCV RANSAC.

Conceptually:

```text
raw LoFTR matches
       │
       ▼
findHomography(..., RANSAC)
       │
       ▼
geometrically consistent inliers
```

The homography maps:

```text
OHRC pixel coordinates
        ↓
TMC pixel coordinates
```

This provides a geometric model for registration and enables reprojection-error evaluation.

---

# 9. Spatial Uniformity Filtering

A common problem with feature matching is that many matches may cluster around a single highly textured area.

For example:

```text
┌──────────────────────┐
│                      │
│              •••••   │
│              •••••   │
│              •••••   │
│                      │
│                      │
└──────────────────────┘
```

Such a match set may produce a mathematically valid transformation but provide poor image-wide coverage.

The project divides the image into a grid.

The standard batch matcher uses:

```text
8 × 8 grid
maximum 2 matches/cell
minimum confidence = 0.1
```

The highest-confidence matches are retained first.

The system also calculates:

* occupied cells,
* coverage ratio,
* entropy-based uniformity score.

---

# 10. Registration

Once correspondence points are available, the project estimates a final homography and warps the OHRC image into TMC coordinates.

The registration module is:

```text
scripts/register.py
```

It generates three QA products:

### `registered_ohrc.png`

OHRC warped into the destination/reference coordinate system.

### `blend_overlay.png`

A 50/50 image blend between the warped OHRC and TMC image.

### `checkerboard_qa.png`

Alternating blocks from the registered source and reference.

The checkerboard is especially useful for visually inspecting:

* crater rim continuity,
* ridge alignment,
* boundary displacement,
* local registration failures.

---

# 11. Evaluation

The evaluation module is:

```text
lunar_project/src/ml/evaluation/metrics.py
```

It provides:

### Number of inliers

```text
N_inliers
```

### Inlier ratio

```text
N_inliers / N_raw
```

### Reprojection error

For a point:

```text
e_i = || H(x_i) - y_i ||
```

### RMSE

```text
RMSE = sqrt(mean(e_i²))
```

### Mean reprojection error

```text
mean(e_i)
```

### Median reprojection error

Useful for reducing sensitivity to individual outliers.

### Maximum error

The worst individual reprojection error.

### Sub-pixel criterion

The implementation considers the registration sub-pixel accurate when:

```text
mean reprojection error < 1 pixel
```

It additionally reports the fraction of points whose individual error is below one pixel.

### Spatial coverage

An 8 × 8 grid can be used to calculate:

```text
occupied cells / total cells
```

for both source and destination points.

The combined coverage score is the geometric mean of the two coverage ratios.

---

# Current Evaluation Result

The repository currently contains an explicit evaluation summary for `region_001`.

| Metric                  |            Value |
| ----------------------- | ---------------: |
| Inliers                 |               10 |
| Raw matches             |               10 |
| Inlier ratio            |             1.00 |
| RMSE                    |        1.7467 px |
| Mean reprojection error |         1.553 px |
| Median error            |        1.4159 px |
| Maximum error           |         3.232 px |
| Sub-pixel criterion     | **Not achieved** |
| Fraction below 1 px     |              30% |
| Source coverage         |            4.69% |
| Destination coverage    |            4.69% |
| Combined coverage       |            4.69% |
| Mean confidence         |           0.4062 |

This result is important because it demonstrates that the repository's evaluation framework reports failures as well as successes. The current stored `region_001` result should **not** be interpreted as universal sub-pixel performance.

---

# 12. Backend

The backend is implemented using **FastAPI**.

Its responsibility is intentionally separated from the scientific processing layer.

The backend:

```text
loads JSON
   │
   ▼
validates data
   │
   ▼
converts pixels → geographic coordinates
   │
   ▼
serves API responses
```

It does not normally rerun the offline preprocessing pipeline when serving stored data.

The main application is:

```text
backend/main.py
```

---

# Backend Data Loading

`backend/data/loader.py` loads:

* processed triplet manifests,
* user triplet manifests,
* match JSON files,
* ML match outputs.

At startup, these are loaded into memory.

The backend enriches match points with:

```text
OHRC pixel
TMC pixel
OHRC lat/lon
TMC lat/lon
confidence
```

It also derives a homography from the available point correspondences.

---

# Backend Endpoints

## Health

```http
GET /health
```

Returns backend status and number of loaded triplets.

## List triplets

```http
GET /triplets
```

Returns all available triplets.

## Get one triplet

```http
GET /triplets/{triplet_id}
```

## Get correspondence data

```http
GET /triplets/{triplet_id}/matches
```

Returns:

* number of matches,
* match points,
* geographic coordinates,
* confidence values,
* homography,
* evaluation metrics.

## Image serving

```http
GET /images/{sensor}/{identifier}
```

The image router supports OHRC, TMC, IIRS and DEM assets and also provides access to registration products.

## Refresh

```http
GET /refresh
```

Reloads the stored data without restarting the backend.

---

# Dynamic Registration API

The backend also exposes:

```http
POST /register
```

with:

```text
source_file
reference_file
```

The uploaded files are checked for:

* supported extension,
* maximum size,
* readable image data,
* maximum image dimension.

Large images are automatically resized to reduce resource usage.

The endpoint then invokes the matching pipeline and returns:

```json
{
  "status": "success",
  "metrics": {},
  "homography": [],
  "visual_url": "...",
  "warped_url": "...",
  "matches_url": "..."
}
```

---

# 13. Frontend

The frontend is a **Next.js 14 / React 18** application.

Dependencies include:

* Next.js
* React
* Leaflet
* React Leaflet
* Three.js
* Tailwind CSS
* TypeScript

The main application provides an interactive lunar archive/inspection console.

---

# Frontend Features

The interface provides three main inspection modes:

```text
FUSED MAP
LINKED CURSOR
REGISTRATION QA
```

The console can:

* browse available regions,
* search tiles,
* inspect triplet metadata,
* view imagery,
* inspect match points,
* inspect geographic footprints,
* compare sensor representations,
* view registration products,
* inspect evaluation metrics,
* launch dynamic registration.

The frontend communicates with FastAPI through:

```text
NEXT_PUBLIC_API_BASE_URL
```

and defaults to:

```text
http://localhost:8000
```

for the main API client.

---

# Repository Structure

```text
.
├── ML_model/
│   ├── matcher.py
│   ├── matches.json
│   ├── ohrc_512.jpeg
│   ├── tmc_512.jpeg
│   └── match_visual.jpg
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── geo.py
│   ├── schemas.py
│   ├── data/
│   │   └── loader.py
│   └── routers/
│       ├── footprint.py
│       ├── images.py
│       ├── matches.py
│       └── triplets.py
│
├── data_preprocessing_pipeline/
│   ├── config/
│   │   └── default.yaml
│   ├── lunar_pipeline/
│   │   ├── config.py
│   │   ├── georef.py
│   │   ├── illumination.py
│   │   ├── ingest.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── scale.py
│   │   ├── sensors.py
│   │   └── tiling.py
│   ├── matches/
│   ├── processed_triplets/
│   ├── scripts/
│   │   ├── batch_generate_triplets.py
│   │   ├── generate_test_pair.py
│   │   ├── make_demo.py
│   │   ├── make_demo_regions.py
│   │   └── select_triplets.py
│   └── triplets.json
│
├── evaluation_output/
│   ├── evaluation_summary.json
│   └── region_001/
│
├── registration_output/
│   ├── region_001/
│   ├── region_002/
│   ├── region_003/
│   ├── region_004/
│   ├── region_005/
│   ├── region_006/
│   └── ...
│
├── scripts/
│   ├── check_iirs_rotation.py
│   ├── distribution.py
│   ├── generate_enhanced_matches.py
│   ├── register.py
│   └── run_loftr_all_regions.py
│
├── lunar_project/
│   └── src/
│       └── ml/
│           └── evaluation/
│
├── lunar-frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── next.config.js
│
├── processed_user/
│   └── matches/
│
├── requirements.txt
├── render.yaml
├── Procfile
└── README.md
```

---

# Data Products

Each processed triplet generally contains:

```text
region_001/
│
├── ohrc_512.png
├── tmc_512.png
├── iirs_512.png
│
├── dem_512.png
│
├── ohrc_512_census.png
├── tmc_512_census.png
│
├── ohrc_512_gradient.png
├── tmc_512_gradient.png
│
├── ohrc_512_lbp.png
├── tmc_512_lbp.png
│
└── manifest.json
```

The repository contains six numbered regions plus additional triplet examples such as:

```text
triplet_01_ch2_ohr_ncp_202
triplet_new_2022
```

The processed region directories contain the corresponding optical/invariant images and, where available, DEM products.

---

# Match JSON Format

The ML pipeline stores correspondences in a simple JSON array.

Example structure:

```json
[
  {
    "image1_x": 123.4,
    "image1_y": 231.7,
    "image2_x": 97.2,
    "image2_y": 188.9,
    "confidence": 0.82
  }
]
```

The backend converts this into an enriched representation containing:

```text
OHRC pixel
TMC pixel
OHRC geographic coordinate
TMC geographic coordinate
confidence
```

---

# Configuration

The primary preprocessing configuration is:

```text
data_preprocessing_pipeline/config/default.yaml
```

Important parameters include:

```yaml
working_gsd_m: 5.0

tile_size: 1024
overlap: 128

pyramid_levels: 5

illumination:
  photometric_model: lunar_lambert
  shadow_percentile: 3.0
  incidence_shadow_deg: 85.0
  write_invariant: true
  invariant_modes:
    - census
    - gradient
    - lbp

sensors:
  destripe: true
  iirs_reduce: pca
  iirs_pca_components: 1
  iirs_band_index: 0

georef:
  resampling: bilinear
  try_isis: false

output:
  dtype: float32
  compress: lzw
```

---

# Installation

## Clone

```bash
git clone https://github.com/Fable98/chandrayaan2-crossmatch.git
cd chandrayaan2-crossmatch
```

---

# Python Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

---

# Backend Dependencies

Install the root requirements:

```bash
pip install -r requirements.txt
```

The root environment contains the FastAPI/Rasterio/Numpy serving stack.

---

# Preprocessing Dependencies

The scientific preprocessing pipeline has its own dependency set:

```bash
pip install -r data_preprocessing_pipeline/requirements.txt
```

This includes:

* NumPy
* Rasterio
* Affine
* OpenCV
* scikit-image
* pyproj
* PyYAML
* tqdm
* pandas
* Shapely
* GeoPandas

---

# ML Dependencies

The LoFTR matching path additionally requires the relevant PyTorch/Kornia environment.

The matcher imports:

```python
torch
kornia
kornia.feature.LoFTR
```

The required LoFTR model checkpoint must also be supplied.

Expected checkpoint:

```text
loftr_outdoor.ckpt
```

The matcher searches for the checkpoint in the working directory and then alongside `matcher.py`.

**Note:** the checkpoint is not included in the repository itself.

---

# Running the Preprocessing Pipeline

The preprocessing package exposes a CLI through:

```text
data_preprocessing_pipeline/lunar_pipeline/cli.py
```

The pipeline accepts an input directory containing supported products and writes processed products/catalogs into an output directory.

Conceptually:

```bash
python -m lunar_pipeline \
    --input <input_directory> \
    --output <output_directory>
```

Check the CLI help for the exact arguments in the current checkout:

```bash
python -m lunar_pipeline --help
```

The main processing sequence is:

```text
discover_products()
       ↓
open_raster()
       ↓
sensor_cleanup()
       ↓
georeference()
       ↓
photometric normalization
       ↓
CLAHE
       ↓
resample_to_gsd()
       ↓
shadow mask
       ↓
invariant generation
       ↓
Gaussian pyramid
       ↓
write_tiles()
       ↓
catalog.json / catalog.csv
```

This sequence is implemented in `pipeline.py`.

---

# Running Feature Matching

To process all available processed regions:

```bash
python scripts/run_loftr_all_regions.py
```

The script:

1. discovers processed regions,
2. loads OHRC and TMC imagery,
3. runs LoFTR,
4. performs RANSAC,
5. optionally matches invariant representations,
6. applies spatial filtering,
7. computes metrics,
8. writes match JSON files.

Output is written primarily to:

```text
processed_user/matches/
```

---

# Generating Enhanced Matches

The repository also contains:

```bash
python scripts/generate_enhanced_matches.py
```

This script processes:

```text
region_001
region_002
region_003
region_004
region_005
region_006
triplet_01_ch2_ohr_ncp_202
triplet_new_2022
```

It combines existing matches with additional feature-derived correspondences and then generates registration products.

### Important

The current implementation intentionally augments some projected feature correspondences with Gaussian noise before saving them.

Therefore:

> **Enhanced-match output should be clearly distinguished from independently validated raw LoFTR correspondences.**

This is particularly important when presenting scientific accuracy results.

---

# Generating Registration Products

Run:

```bash
python scripts/register.py
```

The script loads:

```text
processed_user/matches/
```

and produces:

```text
registration_output/
```

For each successfully registered region:

```text
region_xxx/
├── registered_ohrc.png
├── blend_overlay.png
└── checkerboard_qa.png
```

---

# Running the Backend

From the repository root:

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend should then be accessible at:

```text
http://localhost:8000
```

FastAPI automatically provides interactive API documentation through its standard documentation routes.

---

# Backend Environment Variables

Useful variables include:

```text
CORS_ORIGIN
ALLOWED_ORIGINS
DATA_DIR
PROCESSED_TRIPLETS_DIR
ML_OUTPUT_DIR
```

The backend supports overriding its data directories without changing source code.

For example:

```bash
export DATA_DIR=/path/to/data
export PROCESSED_TRIPLETS_DIR=/path/to/processed_triplets
export ML_OUTPUT_DIR=/path/to/ML_output
```

---

# Running the Frontend

```bash
cd lunar-frontend
npm install
npm run dev
```

The frontend uses:

```text
NEXT_PUBLIC_API_BASE_URL
```

Example:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Then open the Next.js development server in a browser.

---

# Live Registration

The frontend provides a:

```text
✦ Live Registration
```

workflow.

Users select:

```text
Source Image
Reference Image
```

and submit them to:

```http
POST /register
```

The backend executes the registration pipeline and returns:

```text
RMSE
inlier count
inlier ratio
uniformity
sub-pixel status
```

along with:

```text
registered checkerboard
warped source
match JSON
```

The UI then displays the registration result and QA visualization.

---

# API

## `GET /health`

Example:

```bash
curl http://localhost:8000/health
```

---

## `GET /triplets`

```bash
curl http://localhost:8000/triplets
```

---

## `GET /triplets/{id}`

```bash
curl http://localhost:8000/triplets/region_001
```

---

## `GET /triplets/{id}/matches`

```bash
curl http://localhost:8000/triplets/region_001/matches
```

---

## `GET /images/{sensor}/{identifier}`

Examples:

```text
/images/ohrc/region_001
/images/tmc/region_001
/images/iirs/region_001
/images/dem/region_001
```

---

## `POST /register`

Multipart form:

```text
source_file=<image>
reference_file=<image>
```

Supported uploaded image types:

```text
.jpg
.jpeg
.png
.tif
.tiff
```

The backend enforces a 20 MB upload limit and resizes very large images before matching.

---

# Evaluation Metrics

The project evaluates correspondence quality using several complementary measurements.

| Metric             | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| Inlier count       | Number of geometrically consistent correspondences |
| Inlier ratio       | Fraction surviving RANSAC                          |
| RMSE               | Overall geometric registration error               |
| Mean error         | Average reprojection error                         |
| Median error       | Robust central error                               |
| Maximum error      | Worst-case correspondence                          |
| Fraction < 1 px    | Individual sub-pixel fraction                      |
| Sub-pixel accuracy | Mean error < 1 px                                  |
| Coverage ratio     | Spatial extent of correspondences                  |
| Uniformity         | Distribution quality across image                  |

No single metric should be interpreted in isolation.

For example:

```text
High inlier ratio
+
low RMSE
+
poor spatial coverage
```

can still indicate a weak registration because all correspondences may be concentrated in a small part of the image.

---

# Example Region

`region_001` provides a concrete example of a validated triplet.

Its manifest identifies:

```text
OHRC
ch2_ohr_ncp_20210405t1606536730_d_img_d18

TMC-2
ch2_tmc_ncf_20250807t1904346039_d_img_d18

IIRS
ch2_iir_nri_20211221t0324126144_d_img_hw1
```

with:

```text
OHRC GSD  = 0.25 m
TMC GSD   = 5.4 m
IIRS GSD  = 69.04 m
```

The shared footprint is approximately:

```text
Longitude:
336.484646° → 336.589455°

Latitude:
-3.3748612° → -3.2487328°
```

---

# Important Implementation Notes

## 1. The project uses a shared-bounds geographic model

The production backend does not independently estimate geographic coordinates for every sensor pixel using arbitrary corner rotations.

Instead, processed triplets are assumed to share a common spatial bounding box and 512 × 512 coordinate system.

This is central to the frontend's linked geographic inspection.

---

## 2. There are two matching implementations

The repository contains:

```text
ML_model/matcher.py
```

which implements a more complete pairwise registration pipeline including coordinate rescaling and phase-correlation refinement.

It also contains:

```text
scripts/run_loftr_all_regions.py
```

which implements the batch multi-region workflow.

The two paths should therefore not be described as identical implementations.

---

## 3. Live registration requires the LoFTR checkpoint

`matcher.py` expects:

```text
loftr_outdoor.ckpt
```

but that file is not stored in the repository.

Before using `/register`, ensure the checkpoint is available at one of the paths expected by the matcher.

---

## 4. The backend and scientific pipeline are intentionally separated

The backend mainly performs:

```text
load → validate → enrich → serve
```

rather than:

```text
upload → preprocess → train → match → serve
```

This keeps the web application responsive and avoids recomputing expensive scientific processing for every API request.

---

## 5. Stored evaluation results are not automatically proof of scientific generalization

The repository includes visualization and evaluation artifacts, but metrics should always be interpreted with respect to:

* region,
* sensor pair,
* number of matches,
* match-generation procedure,
* spatial coverage,
* whether matches are raw or augmented.

In particular, the current enhanced-match script contains synthetic coordinate perturbations and therefore should not be presented as an unbiased validation benchmark.

---

# Limitations

Several limitations remain important for future scientific validation.

### Limited validated correspondence density

The currently stored evaluation summary contains only 10 matches for `region_001`, with only three occupied cells out of 64.

This is insufficient to establish dense, image-wide registration quality.

### Scale differences

OHRC, TMC-2 and especially IIRS have very different spatial resolutions.

Although the common working GSD reduces the problem, aggressive downsampling inevitably removes some high-frequency OHRC information.

### Illumination differences

Photometric normalization can reduce global shading differences but cannot perfectly reconstruct terrain appearance under a different Sun geometry.

### Homography assumption

A single 3 × 3 homography is useful for local registration but does not necessarily capture all geometric distortions caused by:

* sensor geometry,
* terrain relief,
* perspective,
* orbital viewing geometry,
* residual georeferencing error.

### IIRS spectral complexity

Reducing IIRS to a small number of PCA components makes matching computationally easier but discards some spectral information.

### LoFTR dependency

The live matching path requires a compatible LoFTR checkpoint and a suitable PyTorch/Kornia environment.

### Synthetic augmentation

The enhanced matching script generates additional correspondences using projected image features and intentionally perturbs their coordinates.

These correspondences should be treated as augmentation/demo data rather than ground-truth observations.

---

# Future Work

The architecture provides a foundation for a more rigorous production system.

## 1. Multi-sensor matching

Extend the current OHRC ↔ TMC workflow to fully validated:

```text
OHRC ↔ TMC-2
OHRC ↔ IIRS
TMC-2 ↔ IIRS
```

with independently measured accuracy for each pair.

## 2. Non-rigid / terrain-aware registration

Replace or supplement a global homography with:

* affine + local refinement,
* piecewise transformations,
* thin-plate splines,
* DEM-aware projection,
* physically derived camera models.

## 3. Stronger sub-pixel refinement

Investigate:

* normalized cross correlation,
* phase correlation with windowing,
* Lucas-Kanade refinement,
* optical-flow refinement,
* feature-specific local optimization.

## 4. Better physical photometric correction

Move from simplified normalization toward a validated lunar photometric model using:

* calibrated illumination geometry,
* incidence angle,
* emission angle,
* phase angle,
* terrain slope/aspect,
* DEM information.

## 5. Larger validation benchmark

Create a benchmark covering:

* many lunar regions,
* different Sun angles,
* different terrain classes,
* different sensor combinations,
* manually or independently validated control points.

## 6. Confidence calibration

LoFTR confidence should be evaluated against actual correspondence correctness rather than used only as a ranking mechanism.

## 7. Automated quality gates

A production pipeline could reject a registration unless:

```text
inlier ratio      ≥ threshold
RMSE              ≤ threshold
coverage          ≥ threshold
spatial entropy   ≥ threshold
```

This would prevent visually convincing but spatially weak registrations from being accepted automatically.

---

# References

### Chandrayaan-2

Indian Space Research Organisation — Chandrayaan-2 mission and payload documentation.

### LoFTR

Sun, J., Shen, Z., Wang, Y., Bao, H. and Zhou, X.

**LoFTR: Detector-Free Local Feature Matching with Transformers.**

### Lunar photometric correction

The repository implements simplified Lunar-Lambert and Hapke-inspired photometric factors for illumination normalization.

### Computer vision

OpenCV is used for:

* resizing,
* gradient operations,
* feature detection,
* RANSAC homography estimation,
* perspective warping,
* phase correlation.

### Geospatial processing

Rasterio, PyProj, Shapely and GeoPandas are used throughout the preprocessing/geospatial stack.

---

# License

See the repository's license and third-party dependency terms before redistributing scientific data, trained models or derived products.

---

# Project Status

This repository is an integrated prototype combining:

```text
Scientific preprocessing
        +
Cross-sensor feature matching
        +
Geometric registration
        +
Quantitative evaluation
        +
Interactive visualization
        +
Live registration API
```

The strongest aspect of the architecture is the separation between **scientific preprocessing**, **ML correspondence**, **registration/evaluation**, and the **web visualization layer**. This allows the underlying matching methodology to evolve without requiring a complete rewrite of the application.

For scientific claims, use the stored evaluation artifacts and independently validated correspondences rather than relying solely on visual alignment.
