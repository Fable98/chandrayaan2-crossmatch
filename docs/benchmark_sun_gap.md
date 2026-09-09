# Sun-Gap Degradation Curve (orbital evidence, not synthetic)

Fit RMSE vs sun-azimuth gap across 10 measured pairs under current code
(`matcher_cfog` + guided refill; MI for LRO, NCC for TMC). Raw data:
`evaluation_output/sun_gap/sun_gap_curve.json` (local-only run artifact).

| Sun gap (°) | Pair | Raw / Inl | Fit RMSE (px) | Outcome |
|---|---|---|---|---|
| 103.6 | region_006 OHRC→NAC (real CDR) | 24 / 5 | 0.1792 | success, LOW |
| 131.8 | region_001 OHRC→NAC (real CDR) | 32 / 6 | 0.4979 | success, LOW |
| 131.8 | region_003 OHRC→NAC (real CDR) | 27 / 5 | 0.5957 | success, LOW |
| 160.8 | region_001 OHRC→TMC | 41 / 7 | 1.2715 | success, LOW |
| 160.8 | region_002 OHRC→TMC | 43 / 6 | 1.7868 | success, LOW |
| 160.8 | region_003 OHRC→TMC | 44 / 6 | 0.9941 | success, LOW |
| 160.8 | region_004 OHRC→TMC | 39 / 6 | 1.8329 | success, LOW |
| 162.3 | region_005 OHRC→TMC | 26 / 6 | 1.7699 | success, LOW |
| 162.3 | region_006 OHRC→TMC | 37 / 6 | 1.2960 | success, LOW |
| 162.3 | triplet_new_2022 | 49 cand / 0 | FAILED (Gate3 pathological distortion) | clean fail |

## Readout

1. **Monotonic degradation, ~+0.1px per 10° past 100°.** 0.18 → 0.5–0.6 → 1.0–1.8px. This replaces the synthetic-brightness stress tests as the illumination-robustness evidence: moderate robustness, not invariance.
2. **Inlier count is gap-independent (5–7 everywhere).** Density is texture-limited, not illumination-limited — consistent with the four negative density experiments.
3. **Gap alone does not predict failure.** `triplet_new_2022` fails at 162.3° while `region_005/006` succeed at the same gap — local texture and distortion conditioning (Gate3) decide. The gate correctly refuses instead of forcing a fit.

## Caveats

- LRO gaps mix conventions (OHRC PDS4 sun azimuth vs LROC sub-solar azimuth); treat as approximate.
- TMC manifests cluster at two gap values (160.78 / 162.26 across regions — values appear reused at generation); per-region sun metadata should be re-derived from PDS4 labels before citing precisely.
- `triplet_01_ch2_ohr_ncp_202` (45/7 @1.55px) lacks a manifest mismatch value and is excluded from the sorted curve.
