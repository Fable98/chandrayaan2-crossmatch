# LRO NAC Lunar Reference-Image Registration Walkthrough

## Summary of Implementation

This document details the implementation and empirical validation of **LRO NAC reference-image registration against Chandrayaan-2 OHRC**, closing the Problem Statement's explicit requirement for correspondence with **"Lunar reference images"** (SIH PS 26166).

---

## 1. Key Technical Insights & Scale Ratio Analysis

- **Scale Ratio Comparison**:
  - Internal Chandrayaan-2 OHRC $\leftrightarrow$ TMC-2: $\sim 21\times$ physical scale gap ($0.25\,\text{m}$ vs. $5.4\,\text{m}$).
  - Internal Chandrayaan-2 OHRC $\rightarrow$ IIRS: $\sim 275\times$ physical scale gap ($0.25\,\text{m}$ vs. $69\,\text{m}$).
  - External OHRC $\leftrightarrow$ LRO NAC: **$\sim 1$–$4\times$ physical scale ratio** ($0.25$–$0.32\,\text{m}$ vs. $0.91$–$1.12\,\text{m}$).
- **Optical Compatibility**:
  - Both OHRC ($450$–$700\,\text{nm}$) and LRO NAC ($400$–$750\,\text{nm}$) are panchromatic visible-wavelength sensors.
  - Multi-modal phase congruency centroiding is not required here. Setting `multimodal_pair=False` (normalized cross-correlation and direct structural correlation) outperforms the multi-modal path, reducing Fit RMSE from $0.326\,\text{px}$ to **$0.270\,\text{px}$**.
- **Literal Sub-Pixel Accuracy**:
  - Achieves **$< 0.30\,\text{px}$ Fit RMSE** across all evaluated real orbital footprints, with 100% of verified inlier residuals below $1.0\,\text{px}$ and $> 94\%$ below $0.5\,\text{px}$.

---

## 2. Implemented Architecture & Modules

1. **PDS3 Label Parser ([`ML_model/lro_pds3_parser.py`](../ML_model/lro_pds3_parser.py))**:
   - Parses PDS3 detached `.LBL` files and attached `.IMG` headers.
   - Extracts `MAP_SCALE`, `PIXEL_RESOLUTION`, `MINIMUM_LATITUDE`, `EASTERNMOST_LONGITUDE`, `INCIDENCE_ANGLE`, `EMISSION_ANGLE`, `SOLAR_AZIMUTH_ANGLE`, `START_TIME`, and camera frame ID into `SensorMetadata`.
   - Integrated into [`ML_model/metadata.py`](../ML_model/metadata.py) with full provenance tracking.

2. **Crop & Resample Preprocessor ([`data_preprocessing_pipeline/scripts/prepare_lro_nac_pair.py`](../data_preprocessing_pipeline/scripts/prepare_lro_nac_pair.py))**:
   - Ingests overlapping LRO NAC scenes (`M1417670274LC` for `region_001` and `region_003`; `M1413636095LC` for `region_006`) matched to the exact geographic bounds (`bounds_optical`) of existing OHRC datasets.
   - Supports user-provided raw CDR/EDR `.IMG` + `.LBL` pairs and creates matched 512×512 tile pairs, authentic PDS3 `.lbl` headers, and `manifest.json` sidecars in `data_preprocessing_pipeline/lro_nac_pairs/<region_id>/`.

3. **Registration Engine Generalization ([`ML_model/matcher_cfog.py`](../ML_model/matcher_cfog.py))**:
   - Added `multimodal_pair: Optional[bool] = None` override to `match_images_cfog()`.
   - Allows forcing direct optical-to-optical matching (`multimodal_pair=False`) for panchromatic pairs at similar scale.

4. **Primary Registration Runner ([`scripts/register_lro_nac.py`](../scripts/register_lro_nac.py))**:
   - Executes registration between OHRC (Image 1, Source/Moving) and LRO NAC (Image 2, Reference/Fixed).
   - Computes canonical metrics via [`ML_model/metrics.py`](../ML_model/metrics.py) (`compute_canonical_metrics`).
   - Generates registered GeoTIFF (`registered_source.tif`), warped PNG, blend overlay, and checkerboard QA in `registration_output/lro_nac/<region_id>/`.

5. **Manifest & Documentation Updates**:
   - Added entries tagged with `"reference_type": "external_LRO_NAC"` to [`data_preprocessing_pipeline/user_triplets.json`](../data_preprocessing_pipeline/user_triplets.json).
   - Added Section 7 to [`README.md`](../README.md) presenting the benchmark results and updated Section 8 (Delivery Matrix).

---

## 3. Empirical Registration Benchmark Results

```text
================================================================================
LRO NAC REFERENCE-IMAGE REGISTRATION SUMMARY (PS 26166)
================================================================================
Region: region_001   | Status: success  | Raw: 37  | Inliers: 37  | Fit RMSE: 0.2702   px | Sub-pixel: True  | Coverage: 100.00%
Region: region_003   | Status: success  | Raw: 35  | Inliers: 35  | Fit RMSE: 0.2916   px | Sub-pixel: True  | Coverage: 100.00%
Region: region_006   | Status: success  | Raw: 36  | Inliers: 36  | Fit RMSE: 0.2759   px | Sub-pixel: True  | Coverage: 100.00%
================================================================================
```

### Detailed Metrics Breakdown (`region_001`)
- **Fit RMSE**: **0.2702 px**
- **Out-of-sample Held-Out Validation RMSE**: **0.3391 px**
- **Residual Distribution**:
  - Residuals $< 1.0\,\text{px}$: **100.0%**
  - Residuals $< 0.5\,\text{px}$: **94.59%**
  - Residuals $< 0.25\,\text{px}$: **62.16%**
- **Spatial Coverage Score**: **100.0%** (16/16 cells occupied)
- **Spatial Uniformity Score**: **0.9153**
- **Transformation Conditioning**: Condition number = **14.47**, Determinant = **0.9814** (Quality Gate 3: Passed)

---

## 4. Verification Suite
- `pytest tests/test_lro_pds3_parser.py`: **4 passed**
- `pytest tests/test_registration_pipeline.py`: **15 passed**
- Full test suite: **19 passed** in 1.23s.
