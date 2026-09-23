# Empirical Covariance-Calibration Report

## 1. Executive Summary & Probabilistic Calibration Verdict

- **Total Match Trials**: 936
- **Valid Unambiguous Matches**: 886 (94.7%)
- **Flat Peaks (Gated)**: 34 (3.6%)
- **Multimodal Peaks (Gated)**: 13 (1.4%)

### Calibration Verdict

> [!IMPORTANT]
> **Status**: `CONSERVATIVE_BOUNDS`  
> **Probabilistically Calibrated**: `False`  
> **Rationale**: Covariance provides conservative upper bounds (ECE=0.397, s^2=0.03). Nominal variance is wider than empirical error; safe for robust estimation, but technically conservative.

---

## 2. Statistical Chi-Square 2D Coverage Analysis

For 2D Gaussian error vectors $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \Sigma)$, the squared Mahalanobis distance $d_M^2 = \mathbf{e}^T \Sigma^{-1} \mathbf{e}$ follows a $\chi_2^2$ distribution.

### A. Direct Mahalanobis Ellipse Radius Coverage

| Confidence Level | Mahalanobis Bound | Theoretical Coverage ($1 - e^{-c/2}$) | Empirical Coverage Observed | Delta (Empirical - Expected) |
| :--- | :--- | :--- | :--- | :--- |
| **1-Sigma Ellipse** | $d_M \le 1.0$ ($d_M^2 \le 1.0$) | **39.35%** | **96.84%** | +57.49% |
| **2-Sigma Ellipse** | $d_M \le 2.0$ ($d_M^2 \le 4.0$) | **86.47%** | **99.44%** | +12.97% |
| **3-Sigma Ellipse** | $d_M \le 3.0$ ($d_M^2 \le 9.0$) | **98.89%** | **99.77%** | +0.88% |

### B. Standard Normal-Equivalent Quantile Coverage

| Standard Normal Quantile | $\chi_2^2$ Critical Threshold | Expected Probability | Empirical Coverage Observed | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **p = 68.27%** (1-sigma 1D equivalent) | $c = 2.2957$ ($d_M \le 1.515$) | **68.27%** | **98.87%** | +30.60% |
| **p = 95.45%** (2-sigma 1D equivalent) | $c = 6.1801$ ($d_M \le 2.486$) | **95.45%** | **99.55%** | +4.10% |
| **p = 99.73%** (3-sigma 1D equivalent) | $c = 11.8288$ ($d_M \le 3.439$) | **99.73%** | **99.77%** | +0.04% |

---

## 3. Reliability Diagram & Calibration Curve

| Nominal Confidence Level $p$ | $\chi_2^2$ Threshold $c(p)$ | Empirical Coverage $\hat{p}$ | Calibration Gap $|\hat{p} - p|$ |
| :--- | :--- | :--- | :--- |
| 10.0% | 0.211 | 71.44% | 61.44% |
| 20.0% | 0.446 | 85.67% | 65.67% |
| 30.0% | 0.713 | 93.79% | 63.79% |
| 40.0% | 1.022 | 96.84% | 56.84% |
| 50.0% | 1.386 | 98.31% | 48.31% |
| 60.0% | 1.833 | 98.76% | 38.76% |
| 70.0% | 2.408 | 98.87% | 28.87% |
| 80.0% | 3.219 | 99.10% | 19.10% |
| 90.0% | 4.605 | 99.44% | 9.44% |
| 95.0% | 5.991 | 99.55% | 4.55% |

- **Expected Calibration Error (ECE)**: `0.3968` (39.68%)
- **Maximum Calibration Error (MCE)**: `0.6567` (65.67%)
- **Root Mean Squared Calibration Error (RMSCE)**: `0.4531`
- **Empirical Variance Scale Factor (s^2)**: `0.0340`

---

## 4. Peak Failure Rates & Outlier Gating Analysis

| Peak Classification | Match Count | Fraction of Total | Mean Euclidean Error | Peak Behavior & Filtering Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Valid (Sharp Curvature)** | 886 | 94.7% | **0.2737 px** (RMSE=0.5868 px) | Primary peak possesses well-conditioned negative Hessian and low secondary ambiguity. |
| **Flat Peak (Ill-conditioned)** | 34 | 3.6% | **0.3826 px** | Low curvature or condition number $\kappa > 40.0$ (aperture problem); safely rejected. |
| **Multimodal (Ambiguous)** | 13 | 1.4% | **2.0329 px** | Distinct secondary local maximum ratio $R_2 / R_1 > 0.80$; safely rejected. |

### Key Finding on Gating Effectiveness
Gating flat and multimodal peaks successfully isolates ambiguous correspondences. Valid matches demonstrate sharp sub-pixel precision, while rejected peaks exhibit larger average spatial errors and are assigned conservative fallback uncertainties to protect geometric estimators.

---

## 5. Breakdown Across Perturbation Dimensions

| Texture Type | Trials | Valid Count | Mean Error (px) | 2-Sigma Coverage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cratered` | 234 | 219 | 0.3228 | 98.6% | PASS |
| `mare` | 234 | 220 | 0.2629 | 99.1% | PASS |
| `ridge_edge` | 234 | 221 | 0.2676 | 100.0% | PASS |
| `rugged` | 234 | 226 | 0.2426 | 100.0% | PASS |
