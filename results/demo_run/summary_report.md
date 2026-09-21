# Chandrayaan-2 Registration Pipeline — Execution Summary Report

**Execution Timestamp**: 2026-09-21 18:25:39 UTC  
**Total Wall-Clock Runtime**: 5.93 seconds  
**Pipeline Execution Status**: `INSUFFICIENT_CORRESPONDENCES`  

---

## 1. Executive Telemetry Overview

| Parameter | Value | Standard Requirement / Target |
| :--- | :--- | :--- |
| **Source Sensor** | OHRC (~0.25 m/px nominal) | Ultra-high resolution lander assessment |
| **Reference Sensor** | TMC-2 (~5.0 m/px nominal, single NCF view) | Single-view topographic reference mapping |
| **Working Physical Scale** | 5.00 m/px | Common-GSD scale-space normalization |
| **Verified Inlier Correspondences** | **0** | ≥ 4 verified tie-points |
| **Inlier Ratio** | **0.0%** | Robust to out-of-plane parallax |
| **Planar Fit RMSE** | **N/A** *(LOW_CONFIDENCE: N=0, Held-out Validation RMSE: N/A (N<8) px)* | Sub-pixel precision (< 2.0 px) |
| **Absolute Selenodetic RMSE** | **N/A** | Topographically corrected 3D distance |
| **Spatial Uniformity Gate** | **PASSED** | Grid-based NMS (10x10 grid, max 4/cell) |
| **Terrain Relief Compensation** | **ACTIVE** | DEM ray-intersection & piecewise affine |

---

## 2. Generated Product Inventory

All primary registration artifacts have been verified and exported to `C:\Users\rohit\chand2\chandrayaan2-crossmatch\results\demo_run\registration_products`:
- **Registered GeoTIFF Raster**: `registered_source.tif` (resampled with CRS and transform)
- **Overlay Preview**: `registered_preview.png`
- **Checkerboard Diagnostic**: `registered_checkerboard.png` (50px alternating tiles)
- **Extracted Correspondences**: `matches.json` (sub-pixel native coordinates)
- **Canonical Metrics**: `metrics.json`
- **Coordinate Transformation**: `transform.json`
- **Observation Metadata**: `metadata.json`
- **Execution Log**: `pipeline.log`

---

## 3. Baseline & Ablation Benchmark Results

| Method | Inlier Count | Inlier Ratio | Pixel RMSE | Absolute RMSE (m) | Processing Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure SIFT** | 4 | 57.1% | inf px | 0.00 m | 0.13 s |
| **Pure LoFTR (Fallback: NCC)** | 0 | 0.0% | N/A | N/A | 0.06 s |
| **Our Pipeline (No DEM)** | 0 | 0.0% | N/A | N/A | 1.76 s |
| **Our Full Pipeline (CFOG+DEM+Grid NMS)** | 0 | 0.0% | N/A | N/A | 1.30 s |

---

## 4. Photogrammetric & Algorithmic Methodology Reference

For full mathematical derivations of the Frequency-Domain Phase Congruency, Channel Features of Oriented Gradients (CFOG), DEM Ray-Intersection, and Selenodetic 3D RMSE formulations, refer to:
- [`docs/methodology.md`](methodology.md)

*Report automatically compiled by `run_demo.py`.*
