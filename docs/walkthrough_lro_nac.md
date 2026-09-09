# LRO NAC Lunar Reference-Image Registration Walkthrough

## Summary of Implementation

This document details the implementation and empirical validation of **LRO NAC reference-image registration against Chandrayaan-2 OHRC**, closing the Problem Statement's explicit requirement for correspondence with **"Lunar reference images"** (SIH PS 26166).

---

## 1. Key Technical Insights & Scale Ratio Analysis

- **Scale Ratio Comparison**:
  - Internal Chandrayaan-2 OHRC $\leftrightarrow$ TMC-2: $\sim 20\times$ physical scale gap ($0.25\,\text{m}$ vs. $5.4\,\text{m}$).
  - Internal Chandrayaan-2 OHRC $\rightarrow$ IIRS: $\sim 275\times$ physical scale gap ($0.25\,\text{m}$ vs. $69\,\text{m}$).
  - External OHRC $\leftrightarrow$ LRO NAC: **$\sim 3.4$–$3.7\times$ physical scale ratio** ($0.25$–$0.32\,\text{m}$ vs. $0.914$–$1.081\,\text{m}$ manifest natives).
- **Optical Compatibility**:
  - Both OHRC ($450$–$700\,\text{nm}$) and LRO NAC ($400$–$750\,\text{nm}$) are panchromatic visible-wavelength sensors.
  - On synthetic proxies, `multimodal_pair=False` (NCC) won (0.326px → 0.270px). On **real CDRs it finds 0 candidates** under the true ~104–132° sun gap; the multimodal MI path (`multimodal_pair=True`) is required and yields 0.18–0.80px fit.
- **Qualified Sub-Pixel Accuracy (real CDRs)**:
  - **$0.18$–$0.80\,\text{px}$ In-Sample Fit RMSE** with 5 inliers each, `LOW_CONFIDENCE`, 5% canonical 10×10 coverage; **held-out validation not computable** (`insufficient_points_for_holdout`, <8 pts). Fragile by construction — density work is the priority follow-up.

---

## 2. Implemented Architecture & Modules

1. **PDS3 Label Parser ([`ML_model/lro_pds3_parser.py`](../ML_model/lro_pds3_parser.py))**:
   - Parses PDS3 detached `.LBL` files and attached `.IMG` headers.
   - Extracts `MAP_SCALE`, `PIXEL_RESOLUTION`, `MINIMUM_LATITUDE`, `EASTERNMOST_LONGITUDE`, `INCIDENCE_ANGLE`, `EMISSION_ANGLE`, `SOLAR_AZIMUTH_ANGLE`, `START_TIME`, and camera frame ID into `SensorMetadata`.
   - Integrated into [`ML_model/metadata.py`](../ML_model/metadata.py) with full provenance tracking.

2. **Crop & Resample Preprocessor ([`data_preprocessing_pipeline/scripts/prepare_lro_nac_pair.py`](../data_preprocessing_pipeline/scripts/prepare_lro_nac_pair.py))**:
   - Legacy proxy path ingested overlapping scenes (`M1417670274LC` ×2, `M1413636095LC` ×1) — retired.
   - Real-CDR evidence lives in `data_preprocessing_pipeline/lro_nac_real/<region_id>/` (rigorous corner-affine crop from the 52224×5064 CDR, pad-square → 512, detached `.lbl` + `manifest.json` with `reference_provenance=real_downloaded_cdr`). Line direction follows flight node (001/003 node D → north-up wins; 006 node A → south-up wins); OHRC east edge extends past the NAC swath (~66% overlap).

3. **Registration Engine Generalization ([`ML_model/matcher_cfog.py`](../ML_model/matcher_cfog.py))**:
   - Added `multimodal_pair: Optional[bool] = None` override to `match_images_cfog()`.
   - Allows forcing direct optical-to-optical matching (`multimodal_pair=False`) for panchromatic pairs at similar scale.

4. **Primary Registration Runner ([`scripts/register_lro_nac.py`](../scripts/register_lro_nac.py))**:
   - Executes registration between OHRC (Image 1, Source/Moving) and LRO NAC (Image 2, Reference/Fixed).
   - Computes canonical metrics via [`ML_model/metrics.py`](../ML_model/metrics.py) (`compute_canonical_metrics`), tracking both in-sample `fit_rmse_px` and out-of-sample `held_out_validation_rmse_px`.
   - Generates registered GeoTIFF (`registered_source.tif`), warped PNG, blend overlay, and checkerboard QA in `registration_output/lro_nac/<region_id>/`.

5. **Manifest & Documentation Updates**:
   - Added entries tagged with `"reference_type": "external_LRO_NAC"` to [`data_preprocessing_pipeline/user_triplets.json`](../data_preprocessing_pipeline/user_triplets.json) with both Fit and Validation RMSE.
   - Added Section 7 to [`README.md`](../README.md) presenting the benchmark results and updated Section 8 (Delivery Matrix).

---

## 3. Empirical Registration Benchmark Results

### Benchmark Summary Table

| Region ID | OHRC Product ID | LRO NAC Scene ID | Inliers / Raw | In-Sample Fit RMSE | Held-Out Val RMSE | Sub-Pixel ($<1\,\text{px}$) | Spatial Coverage ($10 \times 10$) | Uniformity | Quality Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `region_001` | `ch2_ohr_ncp_20210405t160653` | `M1417670274LC` (real CDR) | 5 / 23 | **0.8045 px** | null (insufficient pts) | **TRUE** ($<1\,\text{px}$) | 5.0% | 0.0135 | LOW_CONFIDENCE |
| `region_003` | `ch2_ohr_ncp_20210405t160653` | `M1417670274LC` (real CDR) | 5 / 27 | **0.5957 px** | null (insufficient pts) | **TRUE** ($<1\,\text{px}$) | 5.0% | 0.0135 | LOW_CONFIDENCE |
| `region_006` | `ch2_ohr_ncp_20220914t083537` | `M1413636095LC` (real CDR) | 5 / 24 | **0.1792 px** | null (insufficient pts) | **TRUE** ($<1\,\text{px}$) | 5.0% | 0.0135 | LOW_CONFIDENCE |

> [!NOTE]
> Retired proxy scores (37/37 @0.2702px, 35/35 @0.2916px, 36/36 @0.2759px, HIGH) came from OHRC-derived synthetic tiles and are removed from tracking. The NCC path that won on proxies finds 0 candidates on real CDRs.

### CLI Runner Output

```text
==================================================================================================
LRO NAC REFERENCE-IMAGE REGISTRATION SUMMARY — REAL CDR (PS 26166)
==================================================================================================
Region: region_001 | Status: success | Raw: 23 | Inliers: 5 | Fit RMSE: 0.8045 px | Val RMSE: null (insufficient pts) | Sub-pixel: True | Coverage: 5.0% (10x10) | Tier: LOW_CONFIDENCE
Region: region_003 | Status: success | Raw: 27 | Inliers: 5 | Fit RMSE: 0.5957 px | Val RMSE: null (insufficient pts) | Sub-pixel: True | Coverage: 5.0% (10x10) | Tier: LOW_CONFIDENCE
Region: region_006 | Status: success | Raw: 24 | Inliers: 5 | Fit RMSE: 0.1792 px | Val RMSE: null (insufficient pts) | Sub-pixel: True | Coverage: 5.0% (10x10) | Tier: LOW_CONFIDENCE
==================================================================================================
```

### Methodological Rigor & Metric Interpretation
1. **Fit RMSE vs. Held-Out Validation RMSE**:
   - `fit_rmse_px` is in-sample on RANSAC inliers (`fit_rmse_is_in_sample=True`).
   - `held_out_validation_rmse_px` is `insufficient_points_for_holdout` in all 3 real regions (5 inliers < 8 minimum) — the top integrity gap, driving the density work item. Never read fit RMSE without inlier count and tier.
   - 5-point / 8-DOF fits can overfit toward zero (006 @0.18px); the LOW tier and 5% coverage flag this explicitly.

2. **Reference Dataset Scope & Provenance**:
   - 3 real-CDR regions across **2 distinct scenes** (`M1417670274LC`, node D, emi 1.7°; `M1413636095LC`, node A, emi 32°). Sun gaps ~104–132° (convention-approximate) — a genuine illumination stress, which is why MI succeeds where NCC finds nothing.
   - Residuals <1px: 60% (001), 100% (003, 006); <0.5px: 60% (001, 003), 100% (006). Absolute RMSE 0.19–0.74m.

### Detailed Metrics Breakdown
- **`region_001`**: Fit RMSE: **0.8045 px** | Val RMSE: **null** | Coverage: 5.0% | Residuals $< 0.5\,\text{px}$: 60.0%
- **`region_003`**: Fit RMSE: **0.5957 px** | Val RMSE: **null** | Coverage: 5.0% | Residuals $< 0.5\,\text{px}$: 60.0%
- **`region_006`**: Fit RMSE: **0.1792 px** | Val RMSE: **null** | Coverage: 5.0% | Residuals $< 0.5\,\text{px}$: 100.0%

---

## 4. Verification Suite
- `pytest tests/test_lro_pds3_parser.py`: **4 passed**
- `pytest tests/test_registration_pipeline.py`: **15 passed**
- Full test suite: **19 passed** in 1.23s.
