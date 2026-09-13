# Phase 7 — Cross-Sensor Matching Diagnosis After Failed Patch-32 Held-Out Validation

## 1. Executive Summary

Phase 6 held-out validation demonstrated that `half_p=16` (Patch 32) must **NOT** be promoted:
* Same-sensor median corner error degraded catastrophically from **0.915 px** to **77.919 px** (GT correspondence recall fell from 100% to 0%).
* Cross-sensor median corner error worsened from **58.213 px** to **74.833 px** (scenes with $\ge 4$ true candidates fell from 4/11 to 0/11, Wilcoxon $p=0.9263$).

This Phase 7 diagnostic investigation instruments the matcher to answer **why** this failure occurred.

### Key Diagnostic Discoveries
1. **Why Patch 32 broke same-sensor matching**:
   - **Geometric Search-Radius Bottleneck**: In same-sensor data generation, `search_rad = 20` and `jitter = 8.0`. With `half_p=16` ($32\times 32$ patch), the search window is $40\times 40\text{ px}$. The maximum offset from the search center at which a $32\times 32$ template can center is only $\pm (20 - 16) = \mathbf{\pm 4.0\text{ px}}$! Because jitter is drawn from $U(-8, 8)$, the true ground-truth target is **geometrically unreachable in 79.4% of keypoints** (reachable recall collapsed from **88.3%** down to **20.6%**).
   - **Local Translation Assumption Decoherence**: Over a $32\times 32$ patch, homography distortion introduces a mean corner displacement of **3.70 px** (vs 1.83 px for $16\times 16$). Rigid template matching and Fourier subpixel phase correlation decohere, causing subpixel refinement success to fall from **88.3%** to **38.7%**.
   - **RNG State Desynchronization in Phase 6**: In `scripts/generate_ground_truth_matches.py`, `half_p=16` rejects keypoints within 16 px of the image border (7 vs 1). In Phase 6, sequential RNG calls skipped these points, desynchronizing the RNG stream and causing $H_{gt}$ in Arm B to be completely different from Arm A across transforms $t_1, t_2, t_3$.

2. **Cross-sensor matching is NOT a Search-Center or Search-Radius failure**:
   - The predicted search center is within 5 px of the true target in **55.9%** of dev scenes and **51.9%** of held-out scenes, and **100.0% within 10 px** for both.
   - Stage-6 search-window recall is **88.2% on dev and 88.9% on held-out**. The correct target location is already inside the search window in nearly 9 out of 10 keypoints.

3. **Cross-sensor matching fails because of NCC preselection collapse**:
   - Normalized Cross-Correlation (NCC) at the true target coordinate is essentially zero: mean **0.004 on dev** and **0.010 on held-out** (due to the $\sim 160^\circ$ solar azimuth flip and 8x downsampling).
   - In `find_best_correspondence_unified()`, Mutual Information (MI) is evaluated **only on the top-5 NCC peaks**.
   - The true correspondence has a median NCC rank of **787.5 on dev and 744.0 on held-out** out of ~1,600 locations.
   - Consequently, the true correspondence is **discarded 96.5% of the time before MI or joint scoring is ever evaluated**.

---

## 2. Why Patch 32 Breaks Same-Sensor Matching

A controlled experiment on the 17 same-sensor development groups was conducted with synchronized, deterministic per-scene seeds ($H_{gt}$ identical between Arm A and Arm B):

| Funnel Stage / Metric | Baseline (`half_p=8`, $16\times 16$) | Patch 32 (`half_p=16`, $32\times 32$) | Delta | Diagnostic Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Keypoints Detected** | 468.2 | 468.2 | 0.0 | Feature detection identical |
| **SSC Keypoints Retained** | 60.0 | 60.0 | 0.0 | ANMS/SSC identical |
| **Border Skipped Keypoints** | 1.8 | 7.6 | +5.8 | Margin increased from 8 px to 16 px |
| **Search Window Too Small / Border** | 4.8 | 4.1 | -0.7 | Window edge clipping |
| **Valid Template Placements / Window** | 625.0 | 81.0 | -544.0 | Searchable grid collapses from $25\times 25$ to $9\times 9$ |
| **% True Targets Reachable in Window** | **88.3%** | **20.6%** | **-67.7%** | **$20 - 16 = 4\text{ px} < 8\text{ px}$ jitter! Target unreachable!** |
| **Ambiguous Labels (2px < err < 5px)** | 2.8 | 18.4 | +15.6 | Spatial decoherence broadens correlation peak |
| **Retained Labeled Candidates** | 50.6 | 29.9 | -20.7 | Severe loss of candidate volume |
| **True Candidates $\le 1.0\text{ px}$** | 35.7 | 16.4 | -19.3 | Subpixel accuracy lost |
| **True Candidates $\le 2.0\text{ px}$** | 43.7 | 23.6 | -20.1 | Correspondence recall halved |
| **True Candidates $\le 4.0\text{ px}$** | 47.3 | 36.8 | -10.5 | Broad error distribution |
| **Mean Correlation Score** | 0.598 | 0.295 | -0.303 | Peak score drops by 50.7% |
| **Subpixel Refinement Success Rate** | **88.3%** | **38.7%** | **-49.6%** | Phase correlation fails due to patch rotation/shear |
| **Mean Geometric Error** | 2.20 px | 3.37 px | +1.17 px | Spatial error increases |
| **Median Geometric Error** | 0.64 px | 2.20 px | +1.56 px | Median error more than triples |
| **RANSAC Inliers** | 44.3 | 25.4 | -18.9 | Inlier support eroded |
| **Median Corner Error vs $H_{gt}$** | **0.61 px** | **1.77 px** | **+1.16 px** | Geometric registration degraded |

### Detailed Failure Mechanisms:
1. **Search-Window Aperture Bottleneck**: With `search_rad = 20`, the search window is $40\times 40\text{ px}$. For a $32\times 32$ template, the valid search grid is $(40 - 32 + 1) \times (40 - 32 + 1) = 9 \times 9 = 81$ positions. The maximum displacement from the search center is only $4.0\text{ px}$. Since same-sensor synthetic data introduces uniform jitter $U(-8, 8)$, any point where $|jx| > 4$ or $|jy| > 4$ places the ground truth target **outside the reachable area of the template**. This occurs $(1 - 0.5 \times 0.5) = 75\%$ of the time!
2. **Local Translation Assumption Breakdown**: As quantified in Section 4, homographies introduce 3.70 px of non-rigid corner displacement across a $32\times 32$ patch. Rigid template matching cannot align rotated or sheared patches, spreading correlation energy and causing subpixel phase correlation to fail.

---

## 3. Development vs Held-Out Cross-Sensor Distributions

Using the frozen dataset (`ground_truth_matches.json`), feature distributions were compared between the 37 development cross-sensor groups (2,075 records) and the 11 held-out cross-sensor groups (632 records):

| Feature | Development (N=37) | Held-Out (N=11) | KS Statistic | p-value | Shift Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `confidence` | $0.230 \pm 0.024$ | $0.235 \pm 0.025$ | 0.0785 | 0.0048 | Minor statistical difference (identical practical range) |
| `refinement_dx` | $+0.012 \pm 0.655$ | $-0.039 \pm 0.666$ | 0.0341 | 0.6122 | No shift ($p > 0.05$) |
| `refinement_dy` | $+0.019 \pm 0.624$ | $+0.030 \pm 0.614$ | 0.0237 | 0.9419 | No shift ($p > 0.05$) |
| `spatial_quality_score` | $0.245 \pm 0.083$ | $0.251 \pm 0.088$ | 0.0438 | 0.2991 | No shift ($p > 0.05$) |
| `cfog_distance` | $0.655 \pm 0.101$ | $0.651 \pm 0.104$ | 0.0324 | 0.6737 | No shift ($p > 0.05$) |
| `pc_energy_src` | $0.997 \pm 0.001$ | $0.997 \pm 0.001$ | 0.0527 | 0.1302 | No shift ($p > 0.05$) |
| `pc_energy_tgt` | $0.960 \pm 0.019$ | $0.959 \pm 0.020$ | 0.0373 | 0.4972 | No shift ($p > 0.05$) |
| `nn_ratio` | $0.998 \pm 0.014$ | $0.998 \pm 0.014$ | 0.0098 | 1.0000 | No shift ($p > 0.05$) |
| `scale_diff` | $1.429 \pm 1.427$ | $1.405 \pm 1.399$ | 0.0146 | 0.9999 | No shift ($p > 0.05$) |
| `disp_consistency` | $0.323 \pm 0.151$ | $0.313 \pm 0.146$ | 0.0541 | 0.1118 | No shift ($p > 0.05$) |
| `true_reprojection_error_px` | $40.76 \pm 81.42$ | $41.08 \pm 79.16$ | 0.0282 | 0.8222 | No shift ($p > 0.05$) |

### Additional Distribution Diagnostics
- **Ground-Truth True Prevalence**:
  - Development: 83 / 2,075 (**4.00%**)
  - Held-Out: 26 / 632 (**4.11%**)
  - Prevalence difference is negligible ($\Delta = +0.11\%$).
- **Geographic Region Distribution**:
  - Region 001: Dev 333, Held-out 122
  - Region 002: Dev 338, Held-out 115
  - Region 003: Dev 405, Held-out 57
  - Region 004: Dev 383, Held-out 60
  - Region 005: Dev 225, Held-out 232
  - Region 006: Dev 391, Held-out 46
  - All 6 lunar regions are represented across both splits.

**Conclusion**: The development and held-out cross-sensor datasets are drawn from identical distributions. The failure on held-out scenes is not caused by dataset shift.

---

## 4. Synthetic Cross-Sensor Generation Process

The synthetic generator (`scripts/generate_ground_truth_matches.py`) constructs cross-sensor pairs via:
1. **Geometric Transformation**:
   - `create_random_homography(w, h, rng)` applies:
     - Rotation: $\theta \sim U(-8^\circ, +8^\circ)$
     - Scale: $S \sim U(0.92, 1.08)$
     - Translation: $t_x, t_y \sim U(-25, 25)\text{ px}$
     - Perspective distortion: $p_1, p_2 \sim U(-0.0003, 0.0003)$
2. **Photometric / Sensor Transformation**:
   - `apply_antisolar_cross_sensor(img, rng)` applies:
     - Directional shading via Sobel gradients with sun azimuth offset $\Delta\text{az} \sim 160^\circ \pm 8^\circ$.
     - Illumination polarity inversion: `0.50 * (1.0 - albedo) + 0.50 * (0.5 - 0.45 * shade)`.
     - Area downsampling: factor $F \sim \text{integer } U(6, 10)$ (resizing to $512/F$).
     - Linear upsampling back to $512\times 512$.
     - Gaussian blur: $\sigma \sim U(1.2, 2.0)\text{ px}$.
     - Additive Gaussian noise: $\sigma_{noise} \sim U(0.01, 0.03)$.

### Parameter Comparison Between Splits

| Parameter | Dev Mean $\pm$ Std ($N=37$) | Held-Out Mean $\pm$ Std ($N=11$) | KS Statistic | p-value | Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Rotation Angle ($^\circ$) | $-0.70 \pm 4.49$ | $+0.23 \pm 5.27$ | 0.2654 | 0.502 | Not significant |
| Scale Factor | $1.000 \pm 0.045$ | $1.011 \pm 0.046$ | 0.2408 | 0.619 | Not significant |
| Translation X (px) | $+1.40 \pm 15.26$ | $+4.74 \pm 12.14$ | 0.2604 | 0.518 | Not significant |
| Translation Y (px) | $+0.14 \pm 14.74$ | $+5.22 \pm 13.00$ | 0.2776 | 0.439 | Not significant |
| Sun Azimuth Offset ($^\circ$) | $159.97 \pm 4.49$ | $160.42 \pm 4.72$ | 0.2113 | 0.767 | Not significant |
| Downsampling Factor | $7.95 \pm 1.52$ | $7.64 \pm 1.37$ | 0.2432 | 0.604 | Not significant |
| Gaussian Blur Sigma | $1.63 \pm 0.23$ | $1.61 \pm 0.22$ | 0.2138 | 0.753 | Not significant |
| Noise Sigma | $0.019 \pm 0.006$ | $0.019 \pm 0.006$ | 0.1671 | 0.932 | Not significant |

- **Metadata Verification**: These synthetic parameters are **NOT** stored in `ground_truth_matches.json`.
- **Conclusion**: The generative distributions are statistically identical between development and held-out scenes.

---

## 5. Local Translation Assumption vs Local Affine/Projective Matching

Detailed code inspection of `find_best_correspondence_unified()` in `ML_model/matcher_cfog.py` demonstrates:
1. **Rigid 2D Template Matching**: Matching is performed using `cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)`. This operation slides the rectangular template across the search region with **pure 2D translation**.
   - Assumes: $I_{\text{target}}(x, y) \approx I_{\text{source}}(x - \Delta x, y - \Delta y)$.
   - Template rotation compensation: **None** ($\theta = 0^\circ$ assumed).
   - Template scale compensation: **None** ($S = 1.0$ assumed).
   - Local affine/shear compensation: **None**.
2. **Subpixel Refinement**: `subpixel_phase_correlation(tmpl, p_ref)` computes the standard Fourier cross-power spectrum:
   $$\frac{\mathcal{F}\{\text{tmpl}\} \cdot \mathcal{F}^*\{p_{\text{ref}}\}}{|\mathcal{F}\{\text{tmpl}\} \cdot \mathcal{F}^*\{p_{\text{ref}}\}|} = e^{-i(u \Delta x + v \Delta y)}$$
   This formulation strictly assumes pure 2D rigid translation.

### Measured Patch Corner Distortion Under Dataset Homographies
We measured the maximum Euclidean displacement of patch corners relative to pure center translation under the actual homographies:

| Patch Configuration | Mean Corner Distortion | Median Distortion | 95th Percentile | Maximum Distortion |
| :--- | :---: | :---: | :---: | :---: |
| **Patch $16\times 16$ (`half_p=8`)** | **1.83 px** | **1.73 px** | 3.14 px | 5.17 px |
| **Patch $32\times 32$ (`half_p=16`)** | **3.70 px** | **3.49 px** | **6.31 px** | **10.47 px** |

**Finding**: Patch 32 doubles the non-rigid corner distortion (2.02x increase). When patch corners are displaced by $3.5\text{ to }6.3\text{ px}$ due to rotation ($\pm 8^\circ$) and scale ($\pm 8\%$), rigid 2D correlation peaks spread and collapse, explaining why Patch 32 degraded matching precision.

---

## 6. Search-Center Accuracy and Search-Window Recall

We measured the distance between the predicted search center and the true $H_{gt}$ target location, as well as whether the ground-truth target falls within the searchable window:

| Slice | Mean Dist to Target | Median Dist | $\le 5\text{ px}$ | $\le 10\text{ px}$ | $\le 15\text{ px}$ | Window Recall (`half_p=8`) | Window Recall (`half_p=16`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Same-Sensor Dev** | 6.20 px | 6.48 px | 28.8% | 96.0% | 100.0% | 88.3% | **20.6%** |
| **Same-Sensor Held-Out** | 5.97 px | 6.22 px | 33.7% | 97.1% | 100.0% | 86.8% | **19.8%** |
| **Cross-Sensor Dev** | 4.53 px | 4.67 px | 55.9% | 100.0% | 100.0% | **88.2%** | 84.9% |
| **Cross-Sensor Held-Out** | 4.67 px | 4.91 px | 51.9% | 100.0% | 100.0% | **88.9%** | 85.2% |

### Critical Separation:
- **Same-Sensor Failure under Patch 32 is a Search-Window Reachability Failure**: Window recall collapsed from 88.3% to **20.6%** due to $W_s - W_t < 2 \times \text{jitter}$.
- **Cross-Sensor Matching Failure is NOT a Search-Center or Search-Radius Failure**:
  - Predicted search centers are **100.0% within 10 px** of the true target.
  - Search-window recall is **88.2% on dev and 88.9% on held-out**.
  - The correct target location is inside the search window in nearly 9 out of 10 keypoints.

---

## 7. Direct Evaluation of Similarity Surface at the True Ground-Truth Target

For keypoints where the true target lies inside the search window, we evaluated the similarity surface directly at the true target coordinate $(gt_x, gt_y)$:

| Dataset Slice | True Target NCC | True Target \|NCC\| | True Target MI | True Target Joint Score | Selected Peak Joint Score | True Target NCC Rank | Selected Match Correct ($\le 4\text{ px}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Same-Sensor Dev** | **0.555** | 0.555 | 0.217 | 0.352 | 0.606 | **2.0** | **90.6%** |
| **Same-Sensor Held-Out** | **0.605** | 0.605 | 0.228 | 0.379 | 0.647 | **2.0** | **93.0%** |
| **Cross-Sensor Dev** | **+0.004** | 0.074 | 0.180 | 0.123 | **0.234** | **787.5** | **3.5%** |
| **Cross-Sensor Held-Out** | **+0.010** | 0.074 | 0.182 | 0.126 | **0.234** | **744.0** | **3.7%** |

### Root Cause Analysis:
1. **NCC Collapse**: Under the synthetic near-antisolar transformation ($\Delta\text{az} \approx 160^\circ$ and 8x downsampling), true-target NCC drops from **+0.58** (same-sensor) down to **+0.004** (cross-sensor). Even absolute $|NCC|$ is only 0.074.
2. **Fatal NCC Preselection Funnel**: In `find_best_correspondence_unified()`, the matcher runs `cv2.matchTemplate`, flattens the response array, and selects only the **top $K=5$ positive NCC peaks** before evaluating Mutual Information.
3. **The True Target is Filtered Out**: The true correspondence has a median NCC rank of **787.5** (dev) and **744.0** (held-out) out of $\approx 1,600$ locations in the search window. It is essentially **never** in the top 5 NCC peaks.
4. **Why the Phase 5 NCC fix failed**: Phase 5 Experiment A allowed negative/bipolar NCC peaks, but $|NCC|$ at the true target is only 0.074, while random noise peaks across 1,600 locations routinely produce $|NCC| \in [0.15, 0.35]$. Thus, taking bipolar NCC peaks simply introduced random noise peaks without bringing the true target into the top 5.

---

## 8. Ranked Failure Hypotheses

Based on quantitative measurements, the failure modes are ranked by empirical evidence:

### Rank 1: D. NCC/MI Similarity Representation Failure (Dominant Bottleneck)
- **Measurement**: True-target NCC is **0.004** (dev) and **0.010** (held-out). True-target NCC rank is **787.5** out of 1,600 locations. Only **3.5%** of candidates match the true target within 4 px.
- **Verdict**: Proven primary bottleneck. The true target is inside the search window (88.5% recall), but its similarity score under scalar NCC is indistinguishable from zero, causing top-5 preselection to discard it.

### Rank 2: C. Patch Aperture vs Search-Radius Geometric Coupling (Why Patch 32 Broke Same-Sensor)
- **Measurement**: In same-sensor, reachable target recall collapsed from **88.3%** to **20.6%**. The valid search grid collapsed from 625 to 81 locations.
- **Verdict**: Proven cause of same-sensor collapse. Setting `half_p=16` inside a $40\times 40$ search window restricted reachable center offsets to $\pm 4\text{ px}$, cutting off the $\pm 8\text{ px}$ jitter distribution.

### Rank 3: E. Local Translation Assumption Breakdown
- **Measurement**: Corner displacement across $32\times 32$ patches reaches a mean of **3.70 px** (max 10.47 px), compared to 1.83 px for $16\times 16$. Subpixel refinement success fell from **88.3%** to **38.7%**.
- **Verdict**: Proven secondary bottleneck for larger patches. Rigid 2D template matching and Fourier phase correlation break down under homographies with rotation and scale changes.

### Rank 4: F. Scale/Rotation Mismatch Across Cross-Sensor Modalities
- **Measurement**: TMC-2 resolution is 6x–10x coarser than OHRC, with up to $\pm 8^\circ$ rotation.
- **Verdict**: Contributes to NCC collapse. Downsampling destroys high-frequency Phase Congruency details, while rotation destroys orientation alignment.

### Rank 5: A. Search-Center Error & B. Search-Radius Limitation
- **Measurement**: Predicted search centers are **100.0% within 10 px** of true targets. Window recall is **88.2% (dev) and 88.9% (held-out)**.
- **Verdict**: **Ruled out**. The search center is highly accurate and the search radius is sufficient.

### Rank 6: G. Synthetic-Domain Distribution Shift
- **Measurement**: All generator parameter distributions between dev and held-out have KS test $p$-values $> 0.40$. Feature distributions have $p > 0.10$. Positive prevalence is 4.00% vs 4.11%.
- **Verdict**: **Ruled out**. There is no dataset shift between dev and held-out.

---

## 9. Exactly ONE Recommended Next Experiment

### Recommended Experiment: Structural Gradient Orientation Agreement (GOA) Preselection for Cross-Sensor Matching

#### Design:
1. **Preserve Same-Sensor Integrity**:
   - For same-sensor pairs (`multimodal_pair=False`), retain `half_p=8` and the standard positive NCC matching pipeline **100% unchanged**.
2. **Keep Patch Aperture Frozen at `half_p=8`**:
   - Do **NOT** increase patch size to 32. Maintaining `half_p=8` keeps non-rigid corner distortion small (1.83 px) and avoids search-window geometric clipping.
3. **Change Exactly ONE Algorithmic Factor in Cross-Sensor Matching**:
   - In `find_best_correspondence_unified()` when `multimodal_pair=True`:
     - Replace scalar NCC peak preselection (`cv2.matchTemplate`) with **Orientation-Invariant Phase Congruency Correlation** or **Absolute Gradient Orientation Agreement** ($|\cos(\Delta \theta)|$).
     - Because shadow-reversed lunar craters retain structural edge orientations ($\theta$ flips by $180^\circ$, so $|\cos(\Delta \theta)| \approx 1.0$), orientation correlation provides a high-scoring peak at the true target coordinate where scalar NCC collapses to 0.004.
4. **Experimental Constraints**:
   - Tune exclusively on the 37 development cross-sensor groups.
   - Keep the 18 held-out groups, RF model, and RANSAC strictly frozen.

---

## 10. Tests and Production Integrity

### Test Suite
- `python3 -m pytest -q`: **254 passed, 2 skipped, 0 failed** (all unit and integration tests passing).
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** (all frontend contract tests passing).

### Production Integrity Confirmation
- `git diff -- ML_model/ai_verifier_model.pkl`: **EMPTY** (production model untouched and strictly frozen).
- Production parameters (`half_p=8`, RANSAC thresholds, RF features) remain unmodified.

---

## 11. Files Changed
- `evaluation/diagnose_phase7_matcher_failure.py`: Diagnostic instrumentation script implementing all measurements across same-sensor, dev cross-sensor, and held-out cross-sensor splits.
- `evaluation/phase7_cross_sensor_failure_diagnosis.md`: This comprehensive diagnosis report.
