# Phase 9 — Multi-Scale Structural Information Diagnostic Report

## 1. Objective
Following the rejection of both patch-32 scale-adaptive support (Phase 6 & 7) and Gradient Orientation Agreement (GOA) preselection (Phase 8), this Phase 9 investigation addresses the foundational question:

> Does the true cross-sensor correspondence contain discriminative structural information at larger spatial scales that can separate it from false locations?

The goal is to determine whether expanding the spatial context aperture ($16\times 16 \to 96\times 96$) enables structural Phase Congruency (PC) or Mutual Information (MI) to distinguish true correspondences under synthetic cross-sensor transformations (near-antisolar illumination inversion $\sim 160^\circ$ and 6x–10x area downsampling).

---

## 2. Diagnostic Methodology
1. **Multi-Scale Spatial Supports**: Evaluated 5 spatial supports centered on identical keypoints:
   - $16\times 16$ (`half_p=8`) — Baseline production scale
   - $32\times 32$ (`half_p=16`) — Phase 6/7 scale
   - $48\times 48$ (`half_p=24`) — Moderate context
   - $64\times 64$ (`half_p=32`) — Large context
   - $96\times 96$ (`half_p=48`) — Macro context
2. **Search Grid & Target Sampling**:
   - For each SSC keypoint in the 37 cross-sensor development scenes, the diagnostic search region evaluates a displacement grid of $\pm 20\text{ px}$ (radius $s_{\text{rad}}=20$, jitter $U(-6, 6)$ around $H_{gt}$).
   - At each candidate displacement, template and target patches of size $P \times P$ are cross-correlated.
   - For MI, the joint distribution entropy is evaluated between the source template and target patch at the true target coordinate vs the best false candidate peak (excluding $\pm 2\text{ px}$ of the true target).
3. **Rigorous Evaluability & Boundary Audit**:
   - Tracked valid points and border exclusions across all scales.
4. **Transformation Factor Correlations**:
   - Correlated true-vs-false separation with rotation angle, scale distortion, translation magnitude, area downsampling factor, Gaussian blur sigma, and solar azimuth delta.

---

## 3. Valid-Point Counts and Boundary Audit Across Scales

As spatial aperture increases, image boundaries naturally exclude a growing fraction of keypoints:

| Scale | Evaluable Points | Border-Excluded Points | % of Baseline ($16\times 16$) | Search Grid Dimensions |
| :--- | :---: | :---: | :---: | :---: |
| **$16\times 16$** | **1,745** | 475 | **100.0%** | $41\times 41 = 1,681$ positions |
| **$32\times 32$** | 1,645 | 575 | 94.3% | $41\times 41 = 1,681$ positions |
| **$48\times 48$** | 1,515 | 705 | 86.8% | $41\times 41 = 1,681$ positions |
| **$64\times 64$** | 1,396 | 824 | 80.0% | $41\times 41 = 1,681$ positions |
| **$96\times 96$** | **1,186** | **1,034** | **68.0%** | $41\times 41 = 1,681$ positions |

> [!WARNING]
> In an unmodified $56\times 56$ search window (`search_rad=28`), patches $\ge 64\times 64$ physically cannot fit ($0$ valid placements). In this diagnostic, the target canvas was expanded to evaluate the theoretical discriminability of larger spatial context. Even with canvas expansion, **32.0% of all keypoints are lost to border clipping at $96\times 96$**.

---

## 4. Multi-Scale Structural Discriminability Summary

Evaluated across the 37 development cross-sensor scenes:

| Patch Scale | Valid Points | Median Rank (/ 1,681) | Top-5 Recall | Top-20 Recall | True > Best False (%) | Mean Separation | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$16\times 16$** | 1,745 | **843.0** | 0.23% | 1.15% | **0.06%** | -0.300 | 0.5019 |
| **$32\times 32$** | 1,645 | **747.0** | 0.43% | 1.34% | **0.12%** | -0.152 | 0.5323 |
| **$48\times 48$** | 1,515 | **754.0** | 0.20% | 1.72% | **0.07%** | -0.101 | 0.5404 |
| **$64\times 64$** | 1,396 | **706.5** | 0.50% | 2.08% | **0.14%** | -0.075 | 0.5509 |
| **$96\times 96$** | 1,186 | **672.5** | **0.51%** | **2.45%** | **0.17%** | **-0.049** | **0.5698** |

### Mutual Information (MI) Multi-Scale Separation
| Patch Scale | True MI Mean | Best False MI Mean | MI Mean Separation | True MI > False MI (%) |
| :--- | :---: | :---: | :---: | :---: |
| **$16\times 16$** | 0.180 | 0.185 | -0.005 | 37.0% |
| **$32\times 32$** | 0.054 | 0.057 | -0.003 | 25.8% |
| **$48\times 48$** | 0.026 | 0.027 | -0.002 | 21.1% |
| **$64\times 64$** | 0.015 | 0.016 | -0.001 | 24.5% |
| **$96\times 96$** | 0.007 | 0.007 | -0.000 | 23.0% |

---

## 5. Key Empirical Findings

### 1. The True Target Remains in the Noise Floor Across All Scales
- Median true-target rank sits between **672.5 and 843.0** out of 1,681 positions across all spatial scales. This is essentially the 50th percentile ($1,681 / 2 = 840.5$), representing random noise.
- Top-5 true-target recall stays between **0.20% and 0.51%** across all scales.
- In **> 99.8% of cases**, the best false peak outscores the true target (`True > Best False` $\le 0.17\%$).

### 2. Mutual Information Degrades at Larger Scales
- As patch aperture increases from $16\times 16$ to $96\times 96$, MI values decay from $0.180$ down to $0.007$.
- Larger patches integrate over diverse terrain and shadows, increasing marginal entropy without increasing mutual dependence.
- `True MI > False MI` falls from $37.0\%$ at $16\times 16$ down to $23.0\%$ at $96\times 96$.

### 3. ROC-AUC Confirms Lack of Discrimination
- ROC-AUC at $16\times 16$ is **0.5019** (exact coin flip).
- Increasing spatial support by 36x (to $96\times 96$) only nudges AUC to **0.5698**, which remains far below operational utility (typically AUC $> 0.85$ is required for robust peak selection).

---

## 6. Per-Scene Multi-Scale Analysis (37 Scenes)

Comparing $48\times 48$ vs $16\times 16$ across development scenes:
- Scenes where larger scale improved true-target ranking: **11 / 37 (29.7%)**
- Scenes where larger scale worsened true-target ranking: **1 / 37 (2.7%)**
- Scenes where **NO useful separation existed at either scale**: **23 / 37 (62.2%)**

In nearly two-thirds of the cross-sensor scenes, neither $16\times 16$ nor $48\times 48$ produced any distinct correspondence peak.

---

## 7. Transformation Factor Correlations

Correlating scene-level separation (at $48\times 48$) with generative synthetic parameters:

| Generative Transformation Factor | Mean $\pm$ Std in Dataset | Pearson Correlation ($r$) with Separation | Impact on Correspondence Signal |
| :--- | :---: | :---: | :--- |
| **Area Downsample Factor ($F \in [6, 10]$)** | **$8.05 \pm 1.37$** | **$-0.453$** | **Dominant Destroyer**: Extreme GSD disparity eliminates high-frequency phase alignment |
| **Rotation Magnitude ($|\theta| \le 8^\circ$)** | $4.32^\circ \pm 2.34^\circ$ | **$-0.238$** | Moderate negative impact (rigid patch template mismatch) |
| **Gaussian Blur Sigma ($\sigma \in [1.2, 2.0]$)** | $1.58 \pm 0.24\text{ px}$ | $-0.141$ | Attenuates structural edges |
| **Scale Distortion ($|S - 1.0| \le 0.08$)** | $0.04 \pm 0.02$ | $-0.070$ | Weak negative impact |
| **Translation Magnitude ($|\mathbf{t}| \le 25\text{ px}$)** | $20.24 \pm 6.06\text{ px}$ | $-0.023$ | Negligible impact (search window already covers translation) |
| **Sun Azimuth Offset ($\Delta\text{az} \sim 160^\circ$)** | $161.13^\circ \pm 4.67^\circ$ | $+0.096$ | Binary threshold: all cross-sensor pairs are near-antisolar |

**Crucial Insight**: Area downsampling ($r = -0.453$) is the single strongest factor degrading cross-sensor correspondence. When an image is downsampled by $8\times$, upsampled, and blurred, the structural edge detail that defines local keypoint correspondences is mathematically eradicated.

---

## 8. Official Scientific Classification

### **C — Larger context does not help**

#### Scientific Rationale:
1. **Insignificant Discriminability Gains**: Expanding spatial context from $16\times 16$ to $96\times 96$ reduces median rank only from 843.0 to 672.5 (still deep within the 1,681-element noise floor).
2. **Top-5 Recall Remains Near Zero**: Top-5 recall is $0.23\%$ at $16\times 16$ and only $0.51\%$ at $96\times 96$.
3. **Severe Border Penalties**: 32.0% of keypoints are lost to image boundaries at $96\times 96$, and search windows cannot accommodate large patches without massive runtime inflation.
4. **Data-Generation Realism Gap**: The current synthetic cross-sensor generator combines an extreme 6x–10x downsampling factor with Gaussian blur and a $160^\circ$ antisolar polarity flip. At this level of degradation, local patch-based matching (whether 16 px or 96 px) is mathematically dominated by background noise.

---

## 9. Exactly ONE Recommended Next Direction

### Recommended Next Direction: Macro Closed-Basin Topological Matching / Phase Symmetry Center Correlation

#### Technical Rationale:
1. **Local Gradients Do Not Survive Extreme GSD & Illumination Shifts**: Phases 5, 6, 7, 8, and 9 prove that neither local NCC, local MI, local GOA, nor multi-scale patch expansion can recover true correspondences once high-frequency edge detail is destroyed by 8x downsampling and shadow reversal.
2. **What Actually Survives 8x Downsampling**:
   - Craters remain craters. Even when downsampled by 8x and with reversed illumination, the **topological center of closed basins (Phase Symmetry minima / radial gradient convergence)** is invariant to both scale and illumination direction.
   - Closed-contour circular crater detection (Hough crater transform or Phase Symmetry basin centroiding, already implemented in `detect_blob_centroids`) operates at the macro feature level rather than sliding patch correlation.
3. **Future Step**: Focus algorithmic development on macro feature matching (e.g. matching constellation triangles of closed crater basin centroids) rather than sliding 2D patch correlation.

---

## 10. Tests & Production Artifact Integrity Confirmation

- `python3 -m pytest -q`: **258 passed, 2 skipped, 0 failed** in 38.45s.
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** in 226ms.
- `git diff -- ML_model/ai_verifier_model.pkl`: **EMPTY** (production RF model strictly frozen).
- Production parameters (`half_p=8`, RANSAC iterations, thresholds, RF features, `use_goa_preselection=False`) remain strictly intact.

### Files Created:
- [`evaluation/diagnose_phase9_multiscale.py`](evaluation/diagnose_phase9_multiscale.py): Multi-scale diagnostic measurement script.
- [`evaluation/phase9_multiscale_information_diagnostic.md`](evaluation/phase9_multiscale_information_diagnostic.md): Full Phase 9 diagnostic report.
