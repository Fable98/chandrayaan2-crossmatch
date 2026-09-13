# Phase 6: Held-Out End-to-End Validation of Scale-Adaptive Patch Support

---

## Executive Summary

Scale-adaptive patch support (`half_p=16`, $32 \times 32\text{ px}$ patch) was evaluated in a frozen, leak-free, end-to-end experiment against the production baseline (`half_p=8`, $16 \times 16\text{ px}$ patch) across all **18 held-out test groups** (11 cross-sensor, 7 same-sensor).

### Key Findings
1. **Candidate-Generation Recall Did Not Generalize**: On the 11 unseen cross-sensor scenes, Patch 32 reduced true candidate count from $2.36$ to $0.55$ per scene, dropped candidate precision from $4.07\%$ to $1.31\%$, and reduced scenes with $\ge 4$ true correspondences from $4/11\ (36.4\%)$ to $0/11\ (0.0\%)$.
2. **Cross-Sensor Registration Degraded**: Median corner error worsened by **$-16.62\text{ px}$** (from $58.21\text{ px}$ to $74.83\text{ px}$). In paired per-scene comparison, Patch 32 worsened **8 of 11 scenes** (median paired improvement: **$-33.60\text{ px}$**, Wilcoxon $p = 0.9263$).
3. **Severe Same-Sensor Regression**: On same-sensor held-out scenes, Patch 32 caused a catastrophic failure (median corner error degraded from $0.92\text{ px}$ to $77.92\text{ px}$, true inlier recall collapsed from $100.0\%$ to $0.0\%$). Under projective warping, a $32 \times 32\text{ px}$ rigid patch violates local planarity and translation assumptions.

### Production Recommendation
**DO NOT PROMOTE PATCH 32**

---

## Experimental Setup

* **Test Groups**: 18 deterministically held-out image-pair groups (11 cross-sensor, 7 same-sensor) with zero train/test overlap.
* **Arm A (Production Baseline)**: `half_p = 8` ($16 \times 16\text{ px}$ patch support).
* **Arm B (Patch-32 Experiment)**: `half_p = 16` ($32 \times 32\text{ px}$ patch support).
* **Frozen Models & Invariants**:
  * Random Forest model frozen (trained strictly on 54 training groups; production `ML_model/ai_verifier_model.pkl` untouched).
  * Weighted RANSAC configuration frozen ($2,000$ iterations, $5.0\text{ px}$ threshold, Quality Gate 3).
  * Deterministic random seed ($42$).
  * $H_{gt}$ used strictly as an independent post-hoc evaluation oracle.

---

## Candidate Generation Results (Held-Out Cross-Sensor, N=11)

| Metric | Baseline (`half_p=8`) | Patch 32 (`half_p=16`) | Difference |
| :--- | ---: | ---: | ---: |
| Mean Candidates / Scene | 57.5 | 37.2 | -20.3 |
| True Candidates $\le 1.0\text{ px}$ | 0.09 | 0.00 | -0.09 |
| True Candidates $\le 2.0\text{ px}$ | 0.64 | 0.18 | -0.45 |
| True Candidates $\le 3.0\text{ px}$ | 1.18 | 0.45 | -0.73 |
| True Candidates $\le 4.0\text{ px}$ | 2.36 | 0.55 | -1.82 |
| True Candidates $\le 5.0\text{ px}$ | 2.36 | 0.73 | -1.64 |
| True Candidate Precision ($\le 4.0\text{ px}$) | **4.07%** | **1.31%** | **-2.77%** |
| Mean Geometric Error | 41.08 px | 42.72 px | +1.64 px |
| Median Geometric Error | 17.95 px | 37.93 px | +19.98 px |
| Scenes with $\ge 4$ True Candidates | **4 / 11 (36.4%)** | **0 / 11 (0.0%)** | **-36.4%** |

---

## Cross-Sensor Registration Results (Held-Out Test Set, N=11)

| Metric | Baseline (`half_p=8`) | Patch 32 (`half_p=16`) | Difference |
| :--- | ---: | ---: | ---: |
| Homography Success Rate | 11 / 11 (100.0%) | 11 / 11 (100.0%) | +0.0% |
| Median Inlier Count | 7.0 | 8.0 | +1.0 |
| Median Inlier Ratio | 12.28% | 22.22% | +9.94% |
| Median Reprojection RMSE | 2.104 px | 2.149 px | -0.045 px |
| **Median Corner Error vs $H_{gt}$** | **58.213 px** | **74.833 px** | **-16.620 px (Worse)** |
| GT Correspondence Precision | 0.00% | 0.00% | +0.00% |
| GT Correspondence Recall | 0.00% | 0.00% | +0.00% |

---

## Same-Sensor Registration Results (Held-Out Test Set, N=7)

| Metric | Baseline (`half_p=8`) | Patch 32 (`half_p=16`) | Difference |
| :--- | ---: | ---: | ---: |
| Homography Success Rate | 7 / 7 (100.0%) | 7 / 7 (100.0%) | +0.0% |
| Median Inlier Count | 49.0 | 23.0 | -26.0 |
| Median Inlier Ratio | 84.48% | 76.67% | -7.82% |
| Median Reprojection RMSE | 1.288 px | 1.960 px | -0.672 px |
| **Median Corner Error vs $H_{gt}$** | **0.915 px** | **77.919 px** | **-77.004 px (Severe Regression)** |
| GT Correspondence Precision | 95.92% | 0.00% | -95.92% |
| GT Correspondence Recall | 100.00% | 0.00% | -100.00% |

---

## Paired Cross-Sensor Analysis (Per-Scene Breakdown)

| Group | Baseline Err (px) | Patch32 Err (px) | $\Delta$ Error (px) | Base Inl | P32 Inl | Result |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `region_001:...cross_sensor_t0.png` | 41.18 | 111.81 | -70.63 | 9 | 7 | WORSENED |
| `region_001:...cross_sensor_t3.png` | 280.45 | 63.71 | +216.74 | 6 | 8 | IMPROVED |
| `region_002:...cross_sensor_t0.png` | 58.21 | 69.01 | -10.80 | 7 | 10 | WORSENED |
| `region_002:...cross_sensor_t0.png` | 74.96 | 118.43 | -43.47 | 6 | 11 | WORSENED |
| `region_003:...cross_sensor_t1.png` | 18.61 | 230.79 | -212.18 | 8 | 7 | WORSENED |
| `region_004:...cross_sensor_t3.png` | 79.63 | 74.83 | +4.80 | 7 | 10 | IMPROVED |
| `region_005:...cross_sensor_t1.png` | 25.32 | 65.18 | -39.86 | 6 | 9 | WORSENED |
| `region_005:...cross_sensor_t2.png` | 22.42 | 110.11 | -87.69 | 8 | 6 | WORSENED |
| `region_005:...cross_sensor_t0.png` | 24.58 | 58.18 | -33.60 | 9 | 8 | WORSENED |
| `region_005:...cross_sensor_t2.png` | 89.42 | 52.52 | +36.91 | 7 | 8 | IMPROVED |
| `region_006:...cross_sensor_t2.png` | 68.08 | 75.31 | -7.23 | 8 | 9 | WORSENED |

* **Cases Improved**: 3
* **Cases Worsened**: 8
* **Cases Unchanged**: 0
* **Median Paired Improvement**: **-33.597 px**
* **Mean Paired Improvement**: **-22.456 px**
* **Wilcoxon Signed-Rank Test** ($H_1: \text{Patch32 Error} < \text{Baseline}$): $W = 17.0$, $p = 0.9263$ (no improvement).

---

## Scientific Interpretation

1. **Failure to Generalize**: While increasing patch size to $32 \times 32\text{ px}$ showed promising candidate recall on some training crops, it failed on unseen held-out scenes. On held-out scenes, zero scenes achieved $\ge 4$ true correspondences.
2. **Violation of Local Planarity**: In rugged lunar terrain under non-affine perspective warping, larger template patches suffer severe internal distortion (foreshortening and perspective skew). Rigid 2D cross-correlation assumes pure 2D translation; expanding patch size increases the degree to which non-translational geometric deformation degrades correlation peak sharpness.
3. **Same-Sensor Catastrophe**: The $32 \times 32\text{ px}$ patch severely broke same-sensor registration, collapsing inlier recall from $100\%$ to $0\%$.
4. **Conclusion**: Patch enlargement alone is not a viable solution for cross-sensor registration.
