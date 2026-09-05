# Quantitative Evidence Report

This report is generated from recorded artifacts and explicitly marks unavailable experiments.

## LRO absolute RMSE

```json
{
  "status": "not_available",
  "reason": "No JSON result containing absolute_rmse_m and LRO/basemap provenance was found.",
  "rows": []
}
```

## Ablation

```json
{
  "status": "recorded",
  "source": "C:\\Users\\rohit\\chand2\\chandrayaan2-crossmatch\\evaluation_output\\ablation\\ablation_results.json",
  "results": [
    {
      "method": "Pure SIFT",
      "status": "success",
      "match_count": 7,
      "inlier_count": 4,
      "inlier_ratio": 0.5714,
      "fit_rmse_px": 0.0,
      "absolute_rmse_m": 0.0,
      "runtime_s": 0.1526,
      "homography": [
        [
          -0.39333480226454826,
          0.07876466406724443,
          15.26892005289269
        ],
        [
          -1.4787048571965942,
          -0.6880610131119264,
          362.7103737511122
        ],
        [
          -0.0031201224839890965,
          -0.002853269281418098,
          1.0
        ]
      ]
    },
    {
      "method": "Pure LoFTR (NCC Fallback)",
      "status": "insufficient_matches",
      "match_count": 2,
      "inlier_count": 0,
      "inlier_ratio": 0.0,
      "fit_rmse_px": null,
      "absolute_rmse_m": null,
      "runtime_s": 0.0487,
      "matches": [],
      "failure_analysis": {
        "status": "failure_diagnosed",
        "failure_reason": "insufficient_matches",
        "identified_root_causes": [
          "Geometric verification rejected: insufficient_matches"
        ],
        "primary_root_cause": "Geometric verification rejected: insufficient_matches",
        "diagnostics": {
          "texture_variance": {
            "source_laplacian_var": 589.24,
            "reference_laplacian_var": 61.98
          },
          "illumination": {
            "source_sun_azimuth_deg": 45.0,
            "reference_sun_azimuth_deg": 45.0,
            "delta_azimuth_deg": 0.0
          },
          "scale": {
            "source_gsd_m": 5.0,
            "reference_gsd_m": 5.0,
            "scale_ratio": 1.0
          },
          "dynamic_range": {
            "source_dark_fraction": 0.002,
            "reference_dark_fraction": 0.0,
            "source_saturated_fraction": 0.0,
            "reference_saturated_fraction": 0.0
          }
        },
        "diagnostic_json": "evaluation_output\\ablation\\failure_pure_loftr_(ncc_fallback)\\failure_diagnostic_report.json",
        "diagnostic_image": "evaluation_output\\ablation\\failure_pure_loftr_(ncc_fallback)\\failure_diagnostic_preview.png"
      }
    },
    {
      "method": "Our Pipeline (No DEM)",
      "status": "success",
      "match_count": 77,
      "inlier_count": 6,
      "inlier_ratio": 0.0779,
      "fit_rmse_px": 1.3254,
      "absolute_rmse_m": 7.1571,
      "runtime_s": 1.9179
    },
    {
      "method": "Our Full Pipeline (CFOG+DEM+Grid NMS)",
      "status": "success",
      "match_count": 77,
      "inlier_count": 6,
      "inlier_ratio": 0.0779,
      "fit_rmse_px": 1.3254,
      "absolute_rmse_m": 7.1572,
      "runtime_s": 1.2141
    }
  ],
  "failures": [
    {
      "method": "Pure LoFTR (NCC Fallback)",
      "primary_cause": "Geometric verification rejected: insufficient_matches"
    }
  ]
}
```

## Large-AOI

| Dataset | Pair | Inliers | RMSE (px) | Coverage | Trustworthy |
|---|---|---:|---:|---:|---:|
| region_001 | ohrc_iirs | 7 | 2.1046 | 0.06 | True |
| region_001 | tmc_iirs | 7 | None | None | False |
| region_002 | ohrc_iirs | 7 | None | None | False |
| region_002 | tmc_iirs | 4 | None | None | False |
| region_003 | ohrc_iirs | 8 | None | None | False |
| region_003 | tmc_iirs | 3 | None | None | False |
| region_004 | ohrc_iirs | 5 | None | None | False |
| region_004 | tmc_iirs | 6 | None | None | False |
| region_005 | ohrc_iirs | 8 | None | None | False |
| region_005 | tmc_iirs | 6 | None | None | False |
| region_006 | ohrc_iirs | 4 | None | None | False |
| region_006 | tmc_iirs | 3 | None | None | False |
| triplet_01_ch2_ohr_ncp_202 | ohrc_iirs | 9 | None | None | False |
| triplet_01_ch2_ohr_ncp_202 | tmc_iirs | 6 | None | None | False |
| triplet_new_2022 | ohrc_iirs | 4 | None | None | False |
| triplet_new_2022 | tmc_iirs | 3 | None | None | False |

## IIRS PCA/SAM versus mean collapse

```json
{
  "status": "not_available",
  "reason": "No IIRS hyperspectral .npy/.npz cubes were supplied; PNG projections cannot support this ablation.",
  "rows": []
}
```

## Expanded real-triplet inventory

```json
{
  "status": "recorded",
  "triplet_count": 8,
  "sun_azimuth_mismatch_range_deg": [
    160.77988599999998,
    162.25521400000002
  ],
  "rows": [
    {
      "dataset_id": "region_001",
      "ohrc_sun_azimuth_deg": 269.646098,
      "tmc2_sun_azimuth_deg": 108.866212,
      "sun_azimuth_mismatch_deg": 160.77988599999998,
      "iirs_aoi_km": {
        "width_km": 20.36,
        "height_km": 20.01,
        "detector_pixels_est": "295x290",
        "effective_gsd_m": 39.4316
      }
    },
    {
      "dataset_id": "region_002",
      "ohrc_sun_azimuth_deg": 269.646098,
      "tmc2_sun_azimuth_deg": 108.866212,
      "sun_azimuth_mismatch_deg": 160.77988599999998,
      "iirs_aoi_km": {
        "width_km": 20.36,
        "height_km": 20.01,
        "detector_pixels_est": "295x290",
        "effective_gsd_m": 39.4316
      }
    },
    {
      "dataset_id": "region_003",
      "ohrc_sun_azimuth_deg": 269.646098,
      "tmc2_sun_azimuth_deg": 108.866212,
      "sun_azimuth_mismatch_deg": 160.77988599999998,
      "iirs_aoi_km": {
        "width_km": 20.36,
        "height_km": 20.01,
        "detector_pixels_est": "295x290",
        "effective_gsd_m": 39.4316
      }
    },
    {
      "dataset_id": "region_004",
      "ohrc_sun_azimuth_deg": 269.646098,
      "tmc2_sun_azimuth_deg": 108.866212,
      "sun_azimuth_mismatch_deg": 160.77988599999998,
      "iirs_aoi_km": {
        "width_km": 20.36,
        "height_km": 20.01,
        "detector_pixels_est": "295x290",
        "effective_gsd_m": 39.4316
      }
    },
    {
      "dataset_id": "region_005",
      "ohrc_sun_azimuth_deg": 89.392682,
      "tmc2_sun_azimuth_deg": 251.647896,
      "sun_azimuth_mismatch_deg": 162.25521400000002,
      "iirs_aoi_km": {
        "width_km": 20.29,
        "height_km": 20.01,
        "detector_pixels_est": "285x281",
        "effective_gsd_m": 39.3583
      }
    },
    {
      "dataset_id": "region_006",
      "ohrc_sun_azimuth_deg": 89.392682,
      "tmc2_sun_azimuth_deg": 251.647896,
      "sun_azimuth_mismatch_deg": 162.25521400000002,
      "iirs_aoi_km": {
        "width_km": 20.29,
        "height_km": 20.01,
        "detector_pixels_est": "285x281",
        "effective_gsd_m": 39.3583
      }
    },
    {
      "dataset_id": "triplet_01_ch2_ohr_ncp_202",
      "ohrc_sun_azimuth_deg": 269.646098,
      "tmc2_sun_azimuth_deg": 108.866212,
      "sun_azimuth_mismatch_deg": null,
      "iirs_aoi_km": {
        "width_km": 20.36,
        "height_km": 20.01,
        "detector_pixels_est": "295x290",
        "effective_gsd_m": 39.4316
      }
    },
    {
      "dataset_id": "triplet_new_2022",
      "ohrc_sun_azimuth_deg": 89.392682,
      "tmc2_sun_azimuth_deg": 251.647896,
      "sun_azimuth_mismatch_deg": 162.25521400000002,
      "iirs_aoi_km": {
        "width_km": 20.29,
        "height_km": 20.01,
        "detector_pixels_est": "285x281",
        "effective_gsd_m": 39.3583
      }
    }
  ],
  "note": "Supply a directory of additional processed CH2 triplets and rerun this command.",
  "additional_genuine_triplets_required": true
}
```

## Cross-illumination

```json
{
  "status": "controlled_stress_test",
  "source_pair": "data_preprocessing_pipeline\\processed_triplets\\region_001",
  "rows": [
    {
      "variant": "baseline",
      "status": "geometric_verification_failed",
      "quality_tier": null,
      "inlier_count": null,
      "fit_rmse_px": null,
      "spatial_coverage": null
    },
    {
      "variant": "darkened",
      "status": "geometric_verification_failed",
      "quality_tier": null,
      "inlier_count": null,
      "fit_rmse_px": null,
      "spatial_coverage": null
    },
    {
      "variant": "brightened",
      "status": "geometric_verification_failed",
      "quality_tier": null,
      "inlier_count": null,
      "fit_rmse_px": null,
      "spatial_coverage": null
    },
    {
      "variant": "contrast_reversed",
      "status": "geometric_verification_failed",
      "quality_tier": null,
      "inlier_count": null,
      "fit_rmse_px": null,
      "spatial_coverage": null
    }
  ],
  "note": "Synthetic photometric perturbations are not a substitute for independent sun-angle acquisitions."
}
```
