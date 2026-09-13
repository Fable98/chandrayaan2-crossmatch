# Phase 8 — Isolated Cross-Sensor GOA Preselection Experiment

## 1. Hypothesis
In Phase 7, diagnostic measurements proved that:
- In cross-sensor matching, the true ground-truth target is inside the search window in **88.5%** of cases.
- However, scalar Normalized Cross-Correlation (NCC) at the true target coordinate collapses to near-zero ($+0.004 \pm 0.095$) under near-antisolar ($\sim 160^\circ$) illumination flip and 8x area downsampling.
- Because `find_best_correspondence_unified()` preselects only the top-5 positive NCC peaks, the true correspondence (ranked ~780 out of 1,681 locations) is filtered out before Mutual Information (MI) or joint scoring is ever evaluated.

**Hypothesis**: Replacing scalar NCC preselection with an illumination-polarity-invariant structural metric—**Gradient Orientation Agreement (GOA)**—will place true cross-sensor correspondences into the top-5 candidate pool, giving the downstream MI/joint-scoring stage an opportunity to evaluate them.

---

## 2. Exact GOA Formulation
For template patch $I_1$ and search region $I_2$, edge orientation $\theta$ flips by approximately $180^\circ$ ($\pi$ radians) under shadow reversal. Therefore, $\cos(2\Delta\theta) = \cos(2(\theta_2 - \theta_1))$ is invariant modulo $\pi$:
- When parallel ($\Delta\theta = 0$): $\cos(0) = 1.0 \implies \text{GOA} = 1.0$.
- When antiparallel / shadow-reversed ($\Delta\theta = \pi$): $\cos(2\pi) = 1.0 \implies \text{GOA} = 1.0$.
- When orthogonal ($\Delta\theta = \pi/2$): $\cos(\pi) = -1.0 \implies \text{GOA} = 0.0$.

### Vectorized Sliding Formulation:
Using double-angle trigonometric identities:
$$\cos(2\theta) = \frac{G_x^2 - G_y^2}{G_x^2 + G_y^2} = \frac{G_x^2 - G_y^2}{m^2}, \quad \sin(2\theta) = \frac{2 G_x G_y}{m^2}$$
Weighting each pixel by the gradient magnitude product $w = m_1 m_2$:
$$w \cos(2\Delta\theta) = u_1 u_2 + v_1 v_2$$
where:
$$u_1 = \frac{G_{x1}^2 - G_{y1}^2}{m_1}, \quad v_1 = \frac{2 G_{x1} G_{y1}}{m_1}, \quad u_2 = \frac{G_{x2}^2 - G_{y2}^2}{m_2}, \quad v_2 = \frac{2 G_{x2} G_{y2}}{m_2}$$
The 2D search surface across all valid sliding positions is computed via sliding cross-correlation (`cv2.matchTemplate` with `TM_CCORR`):
$$S_{\text{num}} = \text{corr2D}(u_2, u_1) + \text{corr2D}(v_2, v_1)$$
$$S_{\text{denom}} = \text{corr2D}(m_2, m_1) + 10^{-6}$$
$$\text{GOA}(y, x) = 0.5 \cdot \left(\text{clip}\left(\frac{S_{\text{num}}(y, x)}{S_{\text{denom}}(y, x)}, -1.0, 1.0\right) + 1.0\right) \in [0.0, 1.0]$$

---

## 3. Baseline vs GOA Search-Surface Ranking

Evaluated across all search windows where the ground-truth target lies within the valid search region ($N = 1,940$ keypoints):

| Ranking Metric | Baseline (NCC Top-5) | Experimental (GOA Top-5) | Difference ($\Delta$) | Empirical Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Median True-Target Rank** | **783.0** / 1,681 | **836.0** / 1,681 | **+53.0** | **Worsened** (fell further into noise) |
| **Mean True-Target Rank** | 796.9 / 1,681 | 837.3 / 1,681 | +40.4 | Worsened |
| **Top-1 Recall** | 0.10% | 0.05% | -0.05% | Negligible |
| **Top-3 Recall** | 0.26% | 0.21% | -0.05% | Negligible |
| **Top-5 Recall** | **0.41%** | **0.21%** | **-0.20%** | **Nearly halved** |
| **Top-10 Recall** | 0.82% | 0.41% | -0.41% | Halved |
| **Top-20 Recall** | 1.55% | 1.03% | -0.52% | Reduced |

### Surface Metric Values at True Target vs Selected False Peak
| Metric | True Target Mean $\pm$ Std | Selected False Peak Mean $\pm$ Std | Separation |
| :--- | :---: | :---: | :---: |
| **NCC (Signed)** | $+0.006 \pm 0.095$ | $+0.046 \pm 0.136$ | $+0.040$ |
| **\|NCC\| (Absolute)** | $0.075 \pm 0.059$ | $0.046 \pm 0.136$ | $-0.029$ |
| **GOA (Polarity Invariant)** | $0.530 \pm 0.056$ | **$0.669 \pm 0.047$** | **+0.139 (False peak beats True target!)** |
| **Mutual Information (MI)** | $0.180 \pm 0.019$ | $0.183 \pm 0.019$ | $+0.003$ |
| **Joint Score ($0.6\text{MI} + 0.4\text{NCC}$)** | $0.124 \pm 0.027$ | $0.142 \pm 0.041$ | $+0.018$ |

**Diagnostic Discovery**: At the true target, GOA averages $0.530$ (scarcely above random noise level of $0.500$). Meanwhile, false locations elsewhere in the search window produce accidental orientation alignments averaging $0.669$, consistently beating the true target.

---

## 4. True-Target Top-K Recall

Out of 1,940 searchable ground-truth targets:
* True target is in the **Top-1**: 1 / 1,940 (**0.05%**)
* True target is in the **Top-3**: 4 / 1,940 (**0.21%**)
* True target is in the **Top-5**: 4 / 1,940 (**0.21%**)
* True target is in the **Top-10**: 8 / 1,940 (**0.41%**)
* True target is in the **Top-20**: 20 / 1,940 (**1.03%**)

**Conclusion**: GOA preselection delivers the true target to the downstream MI stage in only **0.21%** of candidate searches (worse than baseline NCC's 0.41%).

---

## 5. Candidate-Generation Metrics on 37 Development Groups

Downstream candidate recovery evaluated across all 37 development cross-sensor scenes:

| Metric | Baseline (NCC Top-5) | GOA Experiment | Delta | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Candidates / Scene** | 53.6 | 53.6 | +0.0 | Identical candidate pool |
| **True Candidates $\le 1.0\text{ px}$** | 0.11 | 0.05 | -0.05 | Degraded |
| **True Candidates $\le 2.0\text{ px}$** | 0.35 | 0.30 | -0.05 | Degraded |
| **True Candidates $\le 3.0\text{ px}$** | 0.81 | 0.59 | -0.22 | Degraded |
| **True Candidates $\le 4.0\text{ px}$** | **1.68** | **1.11** | **-0.57** | **Degraded by 33.9%** |
| **True Candidates $\le 5.0\text{ px}$** | 2.54 | 1.92 | -0.62 | Degraded |
| **Candidate Precision ($\le 4\text{ px}$)** | **3.10%** | **2.10%** | **-1.00%** | Degraded |
| **Mean Geometric Error** | 16.32 px | 17.54 px | +1.21 px | Increased error |
| **Median Geometric Error** | 16.46 px | 17.84 px | +1.38 px | Increased error |
| **Scenes with $\ge 4$ True Candidates** | **4 / 37 (10.8%)** | **1 / 37 (2.7%)** | **-8.1% (Fell from 4 to 1)** | **Critical Failure** |
| **Mean Runtime / Scene** | 0.071s | 0.077s | +0.006s | Low overhead (+8.5%) |

---

## 6. Same-Sensor Regression Check

Evaluated across the 17 development same-sensor groups (854 candidate point matches):
- Total same-sensor candidate tests: **854**
- Total candidate discrepancies: **0**
- Match location delta: **0.000 px**
- Match score delta: **0.000000**
- Same-sensor regression status: **PASSED (100% byte-for-byte identical)**

---

## 7. Runtime Comparison
- Baseline runtime across 37 scenes: **2.627s** (Mean: **0.071s / scene**)
- GOA experiment runtime across 37 scenes: **2.849s** (Mean: **0.077s / scene**)
- Runtime overhead of dense GOA preselection is only **+6 ms / scene** (+8.5%), indicating that the fast sliding CCORR formulation is computationally efficient.

---

## 8. Per-Scene Results (37 Development Cross-Sensor Scenes)

| Group | Base Cands | GOA Cands | Base True $\le 4\text{px}$ | GOA True $\le 4\text{px}$ | Base Med Err | GOA Med Err | Base Runtime | GOA Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `region_001:ohrc_512.png::ohrc_512_cross_sensor_t1.png` | 57 | 57 | 4 | 1 | 15.03px | 18.35px | 0.077s | 0.081s |
| `region_001:ohrc_512.png::ohrc_512_cross_sensor_t2.png` | 56 | 56 | 1 | 2 | 16.71px | 16.70px | 0.075s | 0.079s |
| `region_001:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 54 | 54 | 1 | 1 | 16.79px | 18.12px | 0.073s | 0.077s |
| `region_001:tmc_512.png::tmc_512_cross_sensor_t0.png` | 52 | 52 | 2 | 1 | 17.49px | 19.19px | 0.071s | 0.075s |
| `region_001:tmc_512.png::tmc_512_cross_sensor_t1.png` | 55 | 55 | 0 | 1 | 16.35px | 17.25px | 0.073s | 0.077s |
| `region_001:tmc_512.png::tmc_512_cross_sensor_t2.png` | 45 | 45 | 2 | 2 | 13.65px | 16.96px | 0.067s | 0.071s |
| `region_002:ohrc_512.png::ohrc_512_cross_sensor_t1.png` | 58 | 58 | 2 | 0 | 16.74px | 16.78px | 0.073s | 0.079s |
| `region_002:ohrc_512.png::ohrc_512_cross_sensor_t2.png` | 52 | 52 | 2 | 1 | 15.93px | 19.63px | 0.070s | 0.076s |
| `region_002:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 52 | 52 | 0 | 1 | 18.37px | 16.74px | 0.070s | 0.077s |
| `region_002:tmc_512.png::tmc_512_cross_sensor_t1.png` | 52 | 52 | 2 | 1 | 17.17px | 16.65px | 0.070s | 0.075s |
| `region_002:tmc_512.png::tmc_512_cross_sensor_t2.png` | 57 | 57 | 0 | 1 | 18.50px | 19.68px | 0.072s | 0.081s |
| `region_002:tmc_512.png::tmc_512_cross_sensor_t3.png` | 56 | 56 | 1 | 0 | 16.29px | 17.01px | 0.070s | 0.077s |
| `region_003:ohrc_512.png::ohrc_512_cross_sensor_t0.png` | 55 | 55 | 4 | 2 | 15.47px | 16.55px | 0.072s | 0.078s |
| `region_003:ohrc_512.png::ohrc_512_cross_sensor_t1.png` | 58 | 58 | 1 | 0 | 16.60px | 17.87px | 0.073s | 0.080s |
| `region_003:ohrc_512.png::ohrc_512_cross_sensor_t2.png` | 54 | 54 | 1 | 0 | 18.64px | 19.50px | 0.071s | 0.077s |
| `region_003:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 44 | 44 | 4 | 0 | 17.43px | 18.84px | 0.067s | 0.071s |
| `region_003:tmc_512.png::tmc_512_cross_sensor_t0.png` | 58 | 58 | 2 | 2 | 15.69px | 16.98px | 0.073s | 0.080s |
| `region_003:tmc_512.png::tmc_512_cross_sensor_t2.png` | 58 | 58 | 2 | 0 | 16.08px | 18.01px | 0.072s | 0.078s |
| `region_003:tmc_512.png::tmc_512_cross_sensor_t3.png` | 57 | 57 | 2 | 0 | 16.18px | 17.66px | 0.072s | 0.079s |
| `region_004:ohrc_512.png::ohrc_512_cross_sensor_t0.png` | 59 | 59 | 1 | 1 | 14.93px | 20.15px | 0.074s | 0.082s |
| `region_004:ohrc_512.png::ohrc_512_cross_sensor_t1.png` | 58 | 58 | 3 | 0 | 15.19px | 17.98px | 0.072s | 0.079s |
| `region_004:ohrc_512.png::ohrc_512_cross_sensor_t2.png` | 39 | 39 | 0 | 2 | 16.52px | 14.04px | 0.063s | 0.068s |
| `region_004:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 59 | 59 | 0 | 1 | 18.44px | 17.63px | 0.073s | 0.080s |
| `region_004:tmc_512.png::tmc_512_cross_sensor_t0.png` | 49 | 49 | 2 | 2 | 16.39px | 18.71px | 0.068s | 0.074s |
| `region_004:tmc_512.png::tmc_512_cross_sensor_t1.png` | 58 | 58 | 1 | 4 | 18.27px | 18.06px | 0.073s | 0.079s |
| `region_004:tmc_512.png::tmc_512_cross_sensor_t2.png` | 48 | 48 | 1 | 1 | 16.89px | 17.73px | 0.068s | 0.073s |
| `region_005:ohrc_512.png::ohrc_512_cross_sensor_t0.png` | 57 | 57 | 3 | 1 | 16.42px | 16.89px | 0.072s | 0.079s |
| `region_005:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 58 | 58 | 7 | 2 | 15.36px | 18.33px | 0.073s | 0.080s |
| `region_005:tmc_512.png::tmc_512_cross_sensor_t1.png` | 45 | 45 | 0 | 1 | 18.10px | 18.92px | 0.065s | 0.072s |
| `region_005:tmc_512.png::tmc_512_cross_sensor_t3.png` | 55 | 55 | 1 | 1 | 16.34px | 17.44px | 0.072s | 0.081s |
| `region_006:ohrc_512.png::ohrc_512_cross_sensor_t0.png` | 58 | 58 | 1 | 2 | 15.77px | 19.17px | 0.074s | 0.080s |
| `region_006:ohrc_512.png::ohrc_512_cross_sensor_t1.png` | 56 | 56 | 2 | 1 | 13.36px | 17.78px | 0.073s | 0.079s |
| `region_006:ohrc_512.png::ohrc_512_cross_sensor_t3.png` | 59 | 59 | 2 | 2 | 16.31px | 17.80px | 0.073s | 0.080s |
| `region_006:tmc_512.png::tmc_512_cross_sensor_t0.png` | 54 | 54 | 3 | 0 | 16.87px | 18.55px | 0.071s | 0.077s |
| `region_006:tmc_512.png::tmc_512_cross_sensor_t1.png` | 38 | 38 | 1 | 1 | 13.42px | 17.61px | 0.063s | 0.067s |
| `region_006:tmc_512.png::tmc_512_cross_sensor_t2.png` | 45 | 45 | 0 | 1 | 18.16px | 14.45px | 0.067s | 0.073s |
| `region_006:tmc_512.png::tmc_512_cross_sensor_t3.png` | 58 | 58 | 1 | 2 | 15.05px | 17.28px | 0.074s | 0.081s |

---

## 9. Interpretation: Official Classification

### **C — GOA Rejected**

#### Scientific Rationale:
1. **Failure to Distinguish True Targets**: Under GOA, the median rank of the true ground-truth target is **836.0** out of 1,681 locations—which is slightly worse than baseline NCC (783.0) and virtually indistinguishable from the exact median of random noise ($1,681 / 2 = 840.5$).
2. **False Peak Dominance**: While the true target exhibits an average GOA of $0.530 \pm 0.056$, false locations elsewhere in the search window achieve average GOA scores of $0.669 \pm 0.047$. This occurs because 8x downsampling and Gaussian blur attenuate the true high-frequency edge orientations, while spurious or coincidental alignment with coarse crater features in other parts of the search window easily exceeds $0.60$.
3. **Downstream Degradation**: True candidates $\le 4\text{ px}$ fell from $1.68$ to $1.11$ per scene (-33.9%), and scenes with $\ge 4$ true candidates dropped from **4 / 37 (10.8%) down to 1 / 37 (2.7%)**.
4. **Conclusion**: Gradient Orientation Agreement cannot solve cross-sensor peak preselection. It must **NOT** be promoted or combined with other heuristics.

---

## 10. Recommendation for Next Phase
1. **Do NOT Promote GOA**: Keep `use_goa_preselection=False` in production code.
2. **Do NOT Run Held-Out Registration**: The 18 held-out groups remain frozen and uncompromised.
3. **Algorithmic Root-Cause Insight**:
   - Both scalar NCC (Phase 5 & 7) and Gradient Orientation Agreement (Phase 8) fail at the fine patch level ($16\times 16$) because **individual fine-resolution patch gradients cannot survive an 8x scale gap and a $160^\circ$ solar illumination reversal**.
   - What *does* survive an 8x GSD difference and shadow reversal is not local pixel gradient orientation, but **macro structural topography**: Phase Symmetry of closed crater basins, large-scale ridge lines, and multi-scale Phase Congruency log-polar structures.
   - Future work must address pre-matching scale alignment (e.g. coarse-scale basin matching) rather than local 2D gradient preselection.

---

## 11. Test Results
- `python3 -m pytest tests/test_goa_preselection.py -v`: **4 passed in 0.76s** (verifying polarity invariance, same-sensor safety, determinism, and zero $H_{gt}$ leakage).
- `python3 -m pytest -q`: **258 passed, 2 skipped, 0 failed** in 38.23s.
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** in 228ms.

---

## 12. Production Artifact Integrity Confirmation
- `git diff -- ML_model/ai_verifier_model.pkl`: **EMPTY** (production RF model untouched and strictly frozen).
- Production parameters (`half_p=8`, RANSAC iterations, thresholds, RF feature contract) remain strictly intact.
- Same-sensor matching behavior remains 100% byte-for-byte identical.
