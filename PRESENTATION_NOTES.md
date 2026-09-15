# 🎤 SIH 2026 Presentation Notes

## Key Talking Points

1. **The Problem:** Chandrayaan-2 images have sun-angle variation, ~20× OHRC↔TMC scale gap (~275× to IIRS overlay-only), and 3D topographic relief. Pretrained deep matchers break here because brightness constancy is violated by shadows.

2. **Our Solution:** Deterministic structural pipeline (single-channel Phase Congruency + Fourier phase correlation + Lucas-Kanade) with moderate gain/bias tolerance — explicitly NOT claimed sun-angle invariant (162° diametric reversal yields only fragile LOW fits, never invariance). LoFTR baseline retained for comparison (`ML_model/matcher.py`).

3. **The 8-Phase Pipeline:** Walk through README §8-Phase table. Emphasize honest status: Phase 1 CLAHE+mask done; Phase 2 pyramid built but matching uses Level 0; Phase 3 ~20× via common-GSD resampling (not invariant descriptor); Phase 4 RandomForest is a trained gate, not a pass-through (e9e8998 bundle: 3,980 hand-labeled rows, 9 features — confidence, refinement_dx/dy, spatial_quality_score, cfog_distance, pc_energy_src/tgt, nn_ratio, scale_diff; class-1 recall 0.86→0.87, precision 0.93→0.94 on the 9-vs-4-feature comparison; RANSAC-consensus labels deliberately refused as circular supervision); Phase 5 per-point sub-pixel true, full-scene fit 0.99–1.83 px; Phase 6 Grid NMS + macro-cell + SSC done but density low (6–7 inliers, 6–7% @10×10); Phase 7 weighted-RANSAC live with AI ai_inlier_prob weights (soft weighting by default; hard pre-RANSAC veto at 0.5 active on the production path where MasterRegistrationPipeline forces experimental_stack=True); Phase 8 80/20 held-out implemented (not computable at <8 inliers).

4. **AI-Augmented Photogrammetry:** Supervised ML gate (`ai_verifier.py` RandomForest) trained and live (`is_trained=True`; e9e8998: 3,980 hand-labeled rows, 9 features, class-1 recall 0.86→0.87 / precision 0.93→0.94); Unsupervised ML (PCA-PC1) for IIRS 256→1 reduction is live and used for composed overlay. Robustness comes from signal processing plus a trained verifier that weights RANSAC everywhere and hard-vetoes on the production stack — no black-box hallucinations, and we report the veto as path-dependent rather than claiming it filters on every call path.

5. **Zero Fake Fallbacks:** Four Quality Gates + triplet guard (§2). Never synthesize corners or identity matrices; `triplet_new_2022` fails cleanly on distortion gate. Missing runs = `not_available`, never zero-error.

6. **Results:** Report Fit RMSE alongside inlier count, held-out status, coverage, tier. Example: `region_003` 6 inliers, fit 0.99 px (borderline sub-pixel), held-out N/A, 6% @10×10, LOW tier; LRO real-CDR 5–6 inliers, 0.18–0.60 px fit, held-out N/A, 5–6% coverage, LOW tier. Show checkerboard QA.

## Anticipated Judge Questions

**Q: Why no Deep Learning as primary?**
A: We retained a pretrained LoFTR baseline and evaluated it. Under cross-sensor gaps and shadow reversal the brightness-constancy prior collapses (0 candidates on real LRO CDRs via optical NCC path; MI path needed). Primary is Phase Congruency structural matching with verifiable residuals.

**Q: Where is the AI?**
A: Two places: (1) Supervised RandomForest outlier gate — trained bundle in Phase 4 (e9e8998: 3,980 hand-labeled rows, 9 features; class-1 recall 0.86→0.87, precision 0.93→0.94; RANSAC-consensus labels refused as circular), loads with `is_trained=True`; probabilities weight Phase 7 RANSAC everywhere, and the hard pre-RANSAC veto (0.5) is active by default on the production path (`MasterRegistrationPipeline` forces `experimental_stack=True`, the exact backend call) — we do not overclaim it filters on direct matcher calls where the flag defaults False. Honesty note: `36bc643` later migrated the GT file/training code to synthetic-geometry 11-feature supervision without regenerating the bundle, so the loaded model remains the hand-labeled 9-feature one — flagged, not hidden. (2) Unsupervised PCA for IIRS hyperspectral reduction — live. This is AI-augmented photogrammetry: trained gate + PCA, not end-to-end black box.

**Q: How do you verify the points?**
A: Forward-backward consistency (0.08–0.31 px per-point tracking), RANSAC 5.0 px gate, conditioning/distortion gates, spatial-support gate, and 80/20 held-out validation when ≥8 inliers (currently N/A on primary pairs — reported, not hidden). Canonical fixed 10×10 coverage + entropy uniformity via `metrics.py`.

**Q: What happens on failure?**
A: Clean failure with status codes (`insufficient_correspondences`, `geometric_verification_failed`, distortion rejection, `cycle_not_computable`). Demo: `triplet_new_2022` 162° hardest case (fragile LOW success, null held-out — formerly a clean Gate-3 refusal; both outcomes on record). No identity fallback.
