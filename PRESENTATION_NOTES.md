# 🎤 SIH 2026 Presentation Notes

## Key Talking Points

1. **The Problem:** Chandrayaan-2 images have sun-angle variation, ~20× OHRC↔TMC scale gap (~275× to IIRS overlay-only), and 3D topographic relief. Pretrained deep matchers break here because brightness constancy is violated by shadows.

2. **Our Solution:** Deterministic structural pipeline — descriptor-level illumination invariance via Log-Gabor Phase Congruency (contrast/polarity invariant, Top-Hat pixel mutation OFF by default to preserve crater rims), CFOG descriptors, GOA structural orientation candidate preselection for cross-sensor pairs, Grid NMS, Fourier phase-correlation + Lucas-Kanade sub-pixel refinement, chained homography for the IIRS gap. Precise scope: invariance holds for gain/bias and shadow polarity (proven >0.85 structural repeatability); moved cast-shadow *geometry* at ~162° mismatch still yields only fragile LOW fits, honestly flagged RED — never invariance over geometry. LoFTR baseline retained for comparison (`ML_model/matcher.py`).

3. **The 8-Phase Pipeline:** Walk through README §8-Phase table. Emphasize honest status: Phase 1 CLAHE+mask done; Phase 2 pyramid built but matching uses Level 0; Phase 3 ~20× via common-GSD resampling (not invariant descriptor); Phase 4 RandomForest is an untrained scaffold/pass-through; Phase 5 per-point sub-pixel true, full-scene fit 0.99–1.83 px; Phase 6 Grid NMS + macro-cell + SSC done but density low (6–7 inliers, 6–7% @10×10); Phase 7 weighted-RANSAC mechanism present but effectively standard RANSAC until Phase 4 trained; Phase 8 80/20 held-out implemented (not computable at <8 inliers).

4. **AI-Augmented Photogrammetry:** Supervised ML hook (`ai_verifier.py` RandomForest) wired but untrained; Unsupervised ML (PCA-PC1) for IIRS 256→1 reduction is live and used for composed overlay. Robustness today comes from signal processing; AI augments without black-box hallucinations once trained.

5. **Zero Fake Fallbacks:** Four Quality Gates + triplet guard (§2). Never synthesize corners or identity matrices; `triplet_new_2022` fails cleanly on distortion gate. Missing runs = `not_available`, never zero-error.

6. **Results:** Report Fit RMSE alongside inlier count, held-out status, coverage, tier. Example: `region_003` 6 inliers, fit 0.99 px (borderline sub-pixel), held-out N/A, 6% @10×10, LOW tier; LRO real-CDR 5–6 inliers, 0.18–0.60 px fit, held-out N/A, 5–6% coverage, LOW tier. Show checkerboard QA.

## Anticipated Judge Questions

**Q: Why no Deep Learning as primary?**
A: We retained a pretrained LoFTR baseline and evaluated it. Under cross-sensor gaps and shadow reversal the brightness-constancy prior collapses (0 candidates on real LRO CDRs via optical NCC path; MI path needed). Primary is Phase Congruency structural matching with verifiable residuals.

**Q: Where is the AI?**
A: Two places: (1) Supervised RandomForest outlier gate — interface implemented and called in Phase 4, currently untrained pass-through; needs labelled lunar matches to activate — we do not overclaim it filters today. (2) Unsupervised PCA for IIRS hyperspectral reduction — live. This is AI-augmented photogrammetry: hooks + PCA, not end-to-end black box.

**Q: How do you verify the points?**
A: Objective proof, not eyeballing: 80/20 held-out RMSE (out-of-sample, needs ≥8 inliers), cyclic RMSE on triplet loops, inlier-masked SSIM between reference and warped source, spatial uniformity, and a 0–100 confidence score rendered as a GREEN/YELLOW/RED Traffic Light badge in the UI and `metrics.json`. Forward-backward consistency (0.08–0.31 px per-point tracking), RANSAC 5.0 px gate, conditioning/distortion gates, spatial-support gate on top. Canonical fixed 10×10 coverage + entropy uniformity via `metrics.py`.

**Q: Where do judges see all this?**
A: `metrics.json` per registration (`ssim_score`, `confidence_score`, `traffic_light_color`, `held_out_rmse`, `cyclic_rmse`, `uniformity_score`), the Traffic Light badge in the console, `difference_map.png` (|Ref−Warped|, black = perfect), and `scripts/isro_official_evaluator.py` which aggregates per-pair metrics plus traffic-light distribution into JSON + Markdown summaries.

**Q: What happens on failure?**
A: Clean failure with status codes (`insufficient_correspondences`, `geometric_verification_failed`, distortion rejection, `cycle_not_computable`). Demo: `triplet_new_2022` 162° hardest case (fragile LOW success, null held-out — formerly a clean Gate-3 refusal; both outcomes on record). No identity fallback.
