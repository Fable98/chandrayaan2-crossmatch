# Sun-Gap Degradation Curve (orbital evidence, not synthetic)

Fit RMSE vs sun-azimuth gap across 10 measured pairs under unified matching architecture
(`matcher_cfog` + guided refill; unified MI+NCC for cross-sensor pairs, NCC for unimodal). Raw data:
`evaluation_output/sun_gap/sun_gap_curve.json` (local-only run artifact).

![Accuracy vs Delta Azimuth](accuracy_vs_delta_az.png)

---

## 1. LRO Real CDR Re-benchmark: NCC-Only vs Unified MI+NCC

Empirical re-evaluation conducted on real NASA LRO NAC Calibrated Data Record (CDR) tiles (`region_001`, `region_003`, `region_006`) comparing pure Normalized Cross-Correlation (NCC-only via `--optical-ncc`) against the Unified Mutual Information + NCC similarity surface ($w_{\text{MI}} = 0.6, w_{\text{NCC}} = 0.4$):

| Region | Solar Azimuth Gap Δaz | NCC-Only (`--optical-ncc`) Raw / Inliers | NCC-Only Fit RMSE | Unified MI+NCC Raw / Inliers | Unified MI+NCC Fit RMSE | Outcome |
|---|---|---|---|---|---|---|
| `region_006` | 103.6° | 0 / 0 | N/A (Failed) | 24 / 5 | **0.3014 px** | Sub-pixel success |
| `region_001` | 131.8° | 0 / 0 | N/A (Failed) | 32 / 6 | **0.6333 px** | Sub-pixel success |
| `region_003` | 131.8° | 0 / 0 | N/A (Failed) | 27 / 5 | **1.2860 px** | Registration success |

### Key Finding
- **NCC-Only Failure ($>90^\circ$)**: Because real orbital CDR acquisitions feature sun-azimuth gaps between 103.6° and 131.8°, shadow geometry is reversed across crater floors and walls. Unimodal NCC finds **0 candidates** and suffers complete registration failure.
- **Unified MI+NCC Robustness**: The unified similarity surface scores candidates on joint structural information entropy, finding 24–32 candidate correspondences and achieving sub-pixel accuracy (0.3014 px on `region_006`).

---

## 2. Full Orbital Solar-Gap Degradation Curve

| Sun gap (°) | Pair | Raw / Inl | Fit RMSE (px) | Outcome |
|---|---|---|---|---|
| 103.6 | region_006 OHRC→NAC (real CDR) | 24 / 5 | 0.3014 | success, LOW (sub-pixel) |
| 131.8 | region_001 OHRC→NAC (real CDR) | 32 / 6 | 0.6333 | success, LOW (sub-pixel) |
| 131.8 | region_003 OHRC→NAC (real CDR) | 27 / 5 | 1.2860 | success, LOW |
| 160.8 | region_001 OHRC→TMC | 41 / 7 | 1.2715 | success, LOW |
| 160.8 | region_002 OHRC→TMC | 43 / 6 | 1.7868 | success, LOW |
| 160.8 | region_003 OHRC→TMC | 44 / 6 | 0.9941 | success, LOW |
| 160.8 | region_004 OHRC→TMC | 39 / 6 | 1.8329 | success, LOW |
| 162.3 | region_005 OHRC→TMC | 26 / 6 | 1.7699 | success, LOW |
| 162.3 | region_006 OHRC→TMC | 37 / 6 | 1.2960 | success, LOW |
| 162.3 | triplet_new_2022 | 49 cand / 0 | FAILED (Gate3 pathological distortion) | clean fail |

---

## 3. Readout & Physical Fail-Angle Curve

1. **Monotonic degradation, ~+0.1px per 10° past 100°.** 0.30 → 0.63–1.29 → 1.0–1.8px. This replaces synthetic-brightness stress tests as genuine illumination-robustness evidence: moderate physical robustness, not unphysical invariance.
2. **Fail-Angle Curve (expect fail > 90° for unimodal NCC)**:
   - For unimodal linear correlation (NCC), performance collapses completely at $\Delta\text{az} > 90^\circ$ due to shadow-slope reversal.
   - For multimodal MI+NCC with Phase Congruency and high-pass photometric normalization, valid correspondences survive up to ~162°, where severe geometric distortion triggers Quality Gate 3 safety rejections rather than producing hallucinated registrations.
3. **Inlier count is gap-independent (5–7 everywhere).** Density is texture-limited, not illumination-limited.
4. **Gap alone does not predict failure.** `triplet_new_2022` fails at 162.3° while `region_005/006` succeed at the same gap — local texture and distortion conditioning (Gate3) decide. The gate correctly refuses instead of forcing a fit.

---

## 4. Caveats

- LRO gaps mix conventions (OHRC PDS4 sun azimuth vs LROC sub-solar azimuth); treat as approximate.
- TMC manifests cluster at two gap values (160.78 / 162.26 across regions — values appear reused at generation); per-region sun metadata should be re-derived from PDS4 labels before citing precisely.
- `triplet_01_ch2_ohr_ncp_202` (45/7 @1.55px) lacks a manifest mismatch value and is excluded from the sorted curve.
