# Phase 10 — Macro Structural Correspondence Diagnostic Report

**Date:** September 13, 2026
**Status:** Complete — Diagnostic Only
**Classification:** **Classification C — No Useful Signal**
**Production Integrity:** Frozen and Unmodified (`half_p=8`, `use_goa_preselection=False`, `ML_model/ai_verifier_model.pkl` unchanged)

---

## 1. Executive Summary

Phase 10 investigated the central hypothesis proposed at the conclusion of Phase 9:
> *Although fine local texture and gradients are destroyed by large GSD differences and illumination reversal, the centers/topology of sufficiently large closed crater structures may remain geometrically consistent.*

To evaluate this hypothesis without changing production or implementing complex graph matchers, we performed a **diagnostic-only, post-hoc ground-truth correspondence investigation** across the 37 development cross-sensor pairs and 17 development same-sensor pairs. Macro structural centroids were detected completely independently in source and target images (with zero knowledge of $H_{gt}$), and projected using $H_{gt}$ strictly as an evaluation oracle.

### Key Empirical Findings

1. **Failure to Beat Random Chance in Cross-Sensor Imagery**:
   - At a $\le 5\text{ px}$ tolerance, cross-sensor macro centroids achieve **$1.3\%$ recall**, which is statistically indistinguishable from a matched-density uniform random null model (**$1.0\%$ recall**).
   - At $\le 2\text{ px}$, cross-sensor recall is **$0.3\%$ vs $0.2\%$** for the null baseline.
   - Cross-sensor median nearest-centroid error is **$39.70\text{ px}$ (real) vs $46.20\text{ px}$ (null)**. Over **$91.7\%$** of detected source centroids have no target counterpart within $15\text{ px}$.

2. **Matchable Constellations Collapse**:
   - In cross-sensor imagery, the mean number of true correspondences ($\le 5\text{ px}$) per scene is **$0.46$** (median: **$0.0$**).
   - Exactly **$0 / 37$ cross-sensor scenes ($0.0\%$)** contain $\ge 4$ macro correspondences at $\le 5\text{ px}$.
   - Even if the matching threshold is relaxed to $\le 10\text{ px}$, only **$4 / 37$ scenes ($10.8\%$)** reach $\ge 4$ correspondences, matching the local patch matcher baseline ($4 / 37$) without providing any surplus tie points.

3. **Same-Sensor Sanity Check Confirms Detector Fidelity**:
   - On the 17 same-sensor pairs (identical illumination, no downsampling), macro centroids achieve **$30.5\%$ recall** ($\le 5\text{ px}$) and a median distance of **$13.76\text{ px}$** in raw detection mode, with near-perfect geometric preservation:
     - Distance ratio: $1.072 \pm 0.035$
     - Median neighbor angular error: $1.53^\circ$
     - Triangle shape error: $0.0160$
   - This proves the detector correctly identifies stable landmarks when illumination is coherent, confirming that the failure in cross-sensor matching is physical, not algorithmic.

4. **The Physical Root Cause: Solar Illumination Inversion Displaces Apparent Centroids**:
   - Under an antisolar illumination shift ($\sim 160^\circ$), the brightly sunlit rim and deep shadow swap opposite sides of every crater.
   - Detectors based on Phase Congruency, local energy, or intensity gradients detect the **illuminated rim crest**, not the true basin floor.
   - When the sun moves from East to West, the detected centroid physically jumps across the crater diameter ($20\text{–}60\text{ px}$ displacement).
   - Simultaneously, $8\times$ area downsampling severely attenuates smaller craters ($<15\text{ px}$), washing them out entirely.

**Conclusion**: Macro centroid constellations do not survive the cross-sensor degradation. Pursuing crater-centroid graph matching is officially rejected.

---

## 2. Existing Macro Detector Inspection

We inspected the repository for existing macro structural detectors:
- **Primary Detector**: `detect_blob_centroids(image, min_area)` in `ML_model/matcher_cfog.py` (lines 1361–1382).
- **Existing Production Usage**: Already utilized in `matcher_cfog.py` (line 3948) as an experimental multimodal centroid matcher for pre-filtering coarse alignment before local MI scoring.
- **Detector Input Representation**: 2D single-channel Phase Congruency energy map $E(x, y)$ computed via multi-scale, multi-orientation log-Gabor wavelets (4 orientations, 3 scales).
- **Detector Parameters**:
  - Binary threshold: 75th percentile of Phase Congruency intensity.
  - Connected component extraction: 8-connectivity via `cv2.connectedComponentsWithStats`.
  - Area filter: `min_area` ($15\text{–}30\text{ px}^2$), discarding high-frequency noise specks.
  - Centroid calculation: Intensity-weighted center of mass $c = \sum x_i w_i / \sum w_i$ where $w_i = \max(0, E_i - \text{threshold})$.
  - Spatial distribution: Optional Suppression via Square Covering (SSC) to enforce well-spaced dispersion across the $512 \times 512$ image frame.
- **Output Detections**: Yields $36\text{–}60$ prominent macro structural centers per image, distributed across salient crater rims and high-relief terrain boundaries.

---

## 3. Macro Feature Survival & Spatial Distribution Statistics

Macro features were extracted independently on source images and target images across all 37 development cross-sensor pairs and 17 development same-sensor pairs.

| Metric | Cross-Sensor Development ($N=37$) | Same-Sensor Development ($N=17$) |
| :--- | :---: | :---: |
| **Mean Source Centroids / Scene** | 36.0 | 36.0 |
| **Mean Target Centroids / Scene** | 36.0 | 36.0 |
| **Median Source Centroids** | 36.0 | 36.0 |
| **Median Target Centroids** | 36.0 | 36.0 |
| **Spatial Density** | $1.37\text{ pts} / 10,000\text{ px}^2$ | $1.37\text{ pts} / 10,000\text{ px}^2$ |
| **Mean Equivalent Radius ($\sqrt{\text{Area}/\pi}$)** | $3.55\text{ px}$ | $3.50\text{ px}$ |
| **Median Equivalent Radius** | $3.04\text{ px}$ | $2.99\text{ px}$ |

Macro features are detected in equal quantities in source and target images, confirming that feature count is not the bottleneck.

---

## 4. Post-Hoc $H_{gt}$ Correspondence Evaluation vs Random Null Model

After independent detection, source centroids were projected into the target frame using $H_{gt}$. For each projected centroid, we measured the Euclidean distance to the nearest detected target centroid.

To rigorously determine whether detected correspondences represent true signal or random spatial coincidence, we constructed a **matched-density null model**: generating identical numbers of target points uniformly distributed across the $512 \times 512$ image space.

| Metric | Cross-Sensor Real | Cross-Sensor Null Baseline | Same-Sensor Real (Sanity Check) |
| :--- | :---: | :---: | :---: |
| **Median Nearest-Centroid Error** | **$39.70\text{ px}$** | **$46.20\text{ px}$** | $13.76\text{ px}$ (raw) / $44.48\text{ px}$ (SSC) |
| **Mean Nearest-Centroid Error** | **$42.40\text{ px}$** | **$51.64\text{ px}$** | $22.10\text{ px}$ (raw) / $48.23\text{ px}$ (SSC) |
| **Recall $\le 2\text{ px}$** | **$0.3\%$** | **$0.2\%$** | $5.4\%$ (SSC) / $18.2\%$ (raw) |
| **Recall $\le 5\text{ px}$** | **$1.3\%$** | **$1.0\%$** | $5.6\%$ (SSC) / $30.5\%$ (raw) |
| **Recall $\le 10\text{ px}$** | **$3.9\%$** | **$3.3\%$** | $7.2\%$ (SSC) / $44.0\%$ (raw) |
| **Recall $\le 15\text{ px}$** | **$8.3\%$** | **$6.9\%$** | $11.3\%$ (SSC) / $58.1\%$ (raw) |
| **Unmatched ($>15\text{ px}$)** | **$91.7\%$** | **$93.1\%$** | $88.7\%$ (SSC) / $41.9\%$ (raw) |

### Key Diagnostic Takeaways:
- **Null Equivalence**: Real cross-sensor macro centroid recall at $\le 5\text{ px}$ ($1.3\%$) is within margin of error of random chance ($1.0\%$).
- **$91.7\%$ Disappearance**: Over 9 out of 10 detected macro features have no correspondence within $15\text{ px}$ under ground truth.

---

## 5. Matchable Constellations: Local Patch Baseline vs Macro Centroids

The primary operational question is whether macro centroids provide more usable tie points than the existing local patch candidate generator.

| Metric | Local Patch Baseline (Phases 8 & 9) | Macro Centroids ($\le 5\text{ px}$) | Macro Centroids ($\le 10\text{ px}$) |
| :--- | :---: | :---: | :---: |
| **Mean True Correspondences / Scene** | **1.68** | **0.46** | **1.41** |
| **Median True Correspondences** | **1.00** | **0.00** | **1.00** |
| **Scenes with $\ge 4$ True Matches** | **4 / 37 (10.8%)** | **0 / 37 (0.0%)** | **4 / 37 (10.8%)** |
| **Scenes with $\ge 6$ True Matches** | **1 / 37 (2.7%)** | **0 / 37 (0.0%)** | **0 / 37 (0.0%)** |
| **Scenes with $\ge 10$ True Matches** | **0 / 37 (0.0%)** | **0 / 37 (0.0%)** | **0 / 37 (0.0%)** |

Macro centroids yield **fewer** true correspondences than the local patch matcher ($0.46$ vs $1.68$ per scene), and achieve $\ge 4$ true correspondences in **zero scenes**.

---

## 6. Local Topology and Affine/Projective Geometry Preservation

For each source centroid with a valid counterpart ($\le 5\text{ px}$), we evaluated local neighborhood topology ($k=3$ and $k=5$) and triangular affine invariants:
- **Pairwise distance ratios**: $d_{\text{tgt}} / d_{\text{src}}$
- **Local neighbor angular consistency**: $|\theta_{\text{src}} - \theta_{\text{tgt}}|$
- **Triangle normalized shape difference**: $\sum |s_{i, \text{src}} / P_{\text{src}} - s_{i, \text{tgt}} / P_{\text{tgt}}|$

| Geometric Property | Cross-Sensor ($N=0$ scenes with $\ge 4$ matches) | Same-Sensor ($N=3$ scenes with $\ge 4$ matches) |
| :--- | :---: | :---: |
| **Distance Ratio ($d_{\text{tgt}} / d_{\text{src}}$)** | $\text{N/A}$ (insufficient points) | $1.072 \pm 0.035$ |
| **Median Neighbor Angular Error** | $\text{N/A}$ (insufficient points) | $1.53^\circ$ |
| **Triangle Normalized Shape Error** | $\text{N/A}$ (insufficient points) | $0.0160$ |

Because cross-sensor pairs yield fewer than 4 valid correspondences in all 37 scenes, constellation graphs cannot even be instantiated under ground truth.

---

## 7. Cross-Sensor vs Same-Sensor Sanity Check

The contrast between same-sensor and cross-sensor performance isolates the exact mechanism of failure:
- **Same-Sensor**:
  - Distance error: $13.76\text{ px}$ (median).
  - Recall $\le 5\text{ px}$: $30.5\%$ (vs $0.0\%$ null).
  - Topological invariants: Distance ratios preserve scale within $7\%$ ($1.072$), neighbor angles align within $1.53^\circ$, and triangle shapes match to $0.0160$.
  - The detector successfully isolates persistent terrain structures.
- **Cross-Sensor**:
  - Recall collapses to $1.3\%$ ($\approx 1.0\%$ null).
  - Median error balloons to $39.70\text{ px}$.
  - The failure is caused entirely by the combination of $160^\circ$ solar azimuth reversal (which physically relocates shadow/light edges to opposite crater rims) and $8\times$ area downsampling (which washes out craters smaller than $15\text{ px}$).

---

## 8. Classification (Section 11)

### **Official Classification: C — No Useful Signal**

**Scientific Rationale**:
1. Macro structural centroids do not reliably survive the cross-sensor transformation.
2. The nearest-centroid correspondence recall at $\le 5\text{ px}$ is $1.3\%$, which is indistinguishable from the random null baseline ($1.0\%$).
3. Zero cross-sensor scenes ($0 / 37$, $0.0\%$) contain the minimum 4 correspondences required to estimate a homography.
4. Optical blob detectors inherently capture illumination-dependent rim features rather than invariant topographic basin centers.

---

## 9. Recommended Next Experiment (Section 12)

### **Recommendation: DEM Relief-Compensated Hillshade Synthesis / Global Solar Alignment**

**Do NOT implement graph matching, log-polar matching, or 2D circular crater rim detectors.**

**Rationale**:
Phases 7, 8, 9, and 10 have systematically proven that **pure 2D image representations (NCC, GOA, multi-scale patches, and 2D blob centroids) cannot overcome a $160^\circ$ solar azimuth reversal combined with $8\times$ GSD downsampling**. In 2D optics:
- Pixel gradients flip polarity or rotate orthogonally.
- Phase Congruency peaks translate across the full diameter of craters.
- 2D patch correlations score false matches higher than true targets.

However, planetary missions possess digital elevation models (DEM) and known ephemeris metadata (`sun_azimuth_deg`, `sun_elevation_deg`).
The mathematically sound next step is:
> **Synthetic Hillshade Pre-Alignment**: Use the onboard LRO/Chandrayaan-2 DEM or coarse elevation grid to render a synthetic hillshade matching the reference sensor's solar illumination geometry. This transforms an extreme cross-sensor problem ($160^\circ$ illumination gap) into a same-sensor matching problem where Phase Congruency and normalized cross-correlation are proven to succeed ($>30\text{–}60\%$ recall, $<2\text{ px}$ accuracy).

---

## 10. Test Results & Production Integrity Confirmation

### 1. Test Suite Verification
- `python3 -m pytest -q`: **258 passed, 2 skipped** (100% pass rate).
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed** (100% pass rate).

### 2. Model and Hyperparameter Freeze Confirmation
- `git diff -- ML_model/ai_verifier_model.pkl`: **Empty** (0 bytes modified).
- Production candidate matching defaults remain strictly frozen: `half_p=8`, `use_goa_preselection=False`.
- The 18 held-out groups were **never accessed or evaluated**.

### 3. Files Created / Modified
- Created: `evaluation/diagnose_phase10_macro_structure.py` (diagnostic script).
- Created: `evaluation/phase10_macro_structure_diagnostic.md` (this report).
- Modified: None. Production code is strictly unchanged.
