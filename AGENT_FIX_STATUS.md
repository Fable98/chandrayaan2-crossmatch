# AGENT FIX STATUS — SIH Submission-Readiness Loop

## Environment (Task 0 baseline, 2026-09-14)
- Python: 3.13.7
- Node: v24.12.0
- Branch: main @ 0249228 (Epics 1-3 commit, pushed)
- Frontend smoke (`npm --prefix lunar-frontend run smoke`): 8/8 PASS

## Task 0: Baseline audit
Status: PASS
Baseline:
- Epic tests exist and PASS (7/7):
  - tests/test_epic1_illumination.py (2 passed)
  - tests/test_epic2_candidate_selection.py (2 passed)
  - tests/test_epic3_traffic_light.py (3 passed)
- Frontend smoke: 8/8 PASS.
- Known failing tests (3, all pre-existing on clean main @7ec611a, verified via git-stash A/B):
  1. tests/test_frontend_contract.py::test_generated_ts_contract_matches_openapi — backend-types.ts drifted from openapi.json (TASK 4A).
  2. backend/test_api.py::test_job_manager_thread_safety_all_backends — sqlalchemy circular-import env failure, sqlite store unavailable (TASK 4C).
  3. backend/test_security.py::test_users_db_backend_sqlite — same sqlalchemy env failure, using_database() False (TASK 4B).
- Batched full-suite runs from prior session: all other suites pass (registration_pipeline incl. 20x/IIRS, phase2_pyramid, matcher_magsac, metrics, ai_verifier, ingest, lro parsers, dem ingestion, backend api/security remainder).
Evidence:
- `python -m pytest tests/test_epic1_illumination.py tests/test_epic2_candidate_selection.py tests/test_epic3_traffic_light.py -q` → 7 passed
- `python -m pytest -k "test_generated_ts_contract_matches_openapi or test_job_manager_thread_safety_all_backends or test_users_db_backend_sqlite" -q` → 3 failed (listed above)
- `npm --prefix lunar-frontend run smoke` → pass 8, fail 0

## Task 1: Verify Epic 1, Epic 2, Epic 3 are really fixed
Status: PASS
Changes:
- None (verification only; fixes from commit 0249228 confirmed working).
Tests:
- `python -m pytest tests/test_epic1_illumination.py tests/test_phase1_physics_enforcement.py -v` → 4 passed (Top-Hat disabled by default, pure Log-Gabor PC, physics gates intact)
- `python -m pytest tests/test_epic2_candidate_selection.py tests/test_matcher_magsac.py -v` → 4 passed (GOA default preselection, same-sensor path unchanged)
- `python -m pytest tests/test_epic3_traffic_light.py tests/test_metrics.py -v` → 11 passed (ssim_score/confidence_score/traffic_light_color emitted)
Evidence:
- Epic 1/2/3 targeted runs above, all green
- Empty-metrics honest-failure spot check: `compute_canonical_metrics(empty, None)` → RED / 0 / 0.0 (Task-5 guardrail intact, no forced matrix)

## Task 2: Real region matching (reports/current_region_matching_summary.json)
Status: PASS
Changes:
- `ML_model/matcher_cfog.py::find_best_correspondence_unified`: fixed `ValueError: kth out of bounds` crash when the preselect surface has fewer cells than top_k (region_005 OHRC->TMC hit a 2x2 surface). Top-k selection now falls back to full `argsort` when `k >= flat.size`. Small surfaces now return a scored peak instead of crashing the pipeline.
- `tests/test_epic2_candidate_selection.py::test_epic2_small_search_surface_never_crashes`: regression test (failed before fix with the exact production traceback, passes after).
- `reports/current_region_matching_summary.json`: created via production-style `match_images_cfog(ohrc_512, tmc_512, sensors OHRC->TMC-2, manifest-sidecar GSDs, illumination normalization ON)` for region_001..006.
Tests:
- New crash test + full `tests/test_epic2_candidate_selection.py` + `tests/test_goa_preselection.py` → 7 passed.
Evidence:
- Report: success_count=6/6 (inliers 5-8, fit_rmse 0.0-1.6px) BUT every region is traffic RED (inlier_ratio 0.08-0.11 < 0.30, ssim 0.10-0.30, uniformity ~0.01-0.02; region_004 held_out 27.9px). This matches the repo's documented "fragile LOW" characterization — the engine's Gate-1 (N>=4) passes while the Epic-3 light honestly reports RED. No thresholds were altered to hide this; both signals are recorded per region.
- Honest-limitation note: an early run wrongly passed explicit native GSDs (0.25/5.x) for pre-gridded common-grid tiles, forcing a bogus ~20x resample and 0 correspondences everywhere. Correct invocation lets the manifest sidecar resolve the common grid. Documented here so the mistake is not repeated.

## Task 3: SIH documentation matches current code
Status: PASS
Changes:
- `SIH_SUBMISSION_README.md`: rewrote §1 as descriptor-level illumination invariance (Log-Gabor PC, Top-Hat OFF by default) with Epic-1 evidence; added §6 covering GOA structural preselection + traffic-light objective verification (ssim_score/confidence_score/traffic_light_color, backend + frontend wiring); replaced stale "only region_006 succeeds / 001/002/003/005 fail Gates" limitation with 2026-09-14 Task-2 evidence (6/6 Gate-1 success, 5–8 inliers, all RED); refreshed re-run cheat sheet to the Epic pytest commands.
- `PRESENTATION_NOTES.md`: solution bullet now claims precisely descriptor-level (polarity, not cast-shadow geometry) invariance; judge Q&A gained traffic-light/SSIM/cyclic proof answer plus evaluator + difference-map pointer.
- `tests/test_submission_docs.py`: NEW — asserts README covers all 8 required topics and contains none of the 5 stale claim strings.
Tests:
- `python -m pytest tests/test_submission_docs.py -v` → 2 passed.
Evidence:
- Repo-wide grep: stale strings count 0 in both README and presentation notes (test file itself lists them as forbidden patterns, scoped to README).

## Task 4: Fix the three failing tests
Status: PASS
Changes:
- 4A OpenAPI drift (`test_generated_ts_contract_matches_openapi`): root cause was NOT schema drift — the committed `backend-types.ts` was already in sync. The worktree file carried CRLF (core.autocrlf=true) while `gen-openapi.mjs` canonically emits LF, so the strict byte check failed with no content difference. Fix: `.gitattributes` forces `text eol=lf` for `backend-types.ts` + `gen-openapi.mjs`, and the `--check` comparison normalizes CRLF→LF before comparing (content comparison stays exact; no weakening). Regenerated + committed file is byte-identical to before (git diff empty apart from attributes).
- 4B/4C sqlite tests (`test_users_db_backend_sqlite`, `test_job_manager_thread_safety_all_backends`): root cause was environmental — the local site-packages held a corrupt partial SQLAlchemy tree (no `util/` package, no dist-info; `pip show` = not found) while `requirements.txt` pins `SQLAlchemy==2.0.48`. Fix: `python -m pip install "SQLAlchemy==2.0.48"`. No repo code change; job_store/auth_store locking was already correct (5/5 stable repeats confirm no race).
Tests:
- `python -m pytest tests/test_frontend_contract.py -v` → 3 passed (incl. CRLF round-trip proof: file flipped to CRLF still passes, then restored to LF).
- `python -m pytest -k "test_job_manager_thread_safety_all_backends or test_users_db_backend_sqlite" -v` → 2 passed.
- Thread-safety test ×5 consecutive runs → 5 passed.
Evidence: commands above; `git diff -- lunar-frontend/src/lib/backend-types.ts` empty (no schema change).

## Task 5: Traffic light visible in the frontend UI
Status: PASS
Changes:
- `lunar-frontend/src/components/TrafficLightBadge.tsx`: NEW reusable badge (props color/confidence_score/ssim_score/held_out_rmse). GREEN="Photogrammetric Grade", YELLOW="Acceptable / Review", RED="Rejected / Low Confidence", plus confidence % and SSIM/held-out tooltip. Null/unknown color renders neutral "Unverified" — never crashes, never invents a verdict.
- `lunar-frontend/src/components/RegistrationLauncher.tsx`: `RegistrationMetrics` extended with ssim_score/confidence_score/traffic_light_color/held-out keys; badge rendered in the result header beside the quality-tier pill for every completed run.
- `lunar-frontend/src/components/ingest/ResultsTable.tsx`: badge rendered in each region-bundle expanded row from `matching.*` (fallback `registration.*`).
- `lunar-frontend/scripts/smoke.mjs`: NEW subtest asserting badge labels, null-safety, props, and rendering in both hosts.
Tests:
- `npm --prefix lunar-frontend run smoke` → 9/9 pass (8 prior + badge).
- `npm --prefix lunar-frontend run build` → compiled successfully, types valid, 6/6 static pages.
Evidence: commands above.

## Task 6: Difference-map artifact generation
Status: PASS
Changes:
- None needed — `match_images_cfog` D3 path already writes `difference_map.png` + `metrics["difference_map"]` + `outputs["difference_map"]` on success. Added proof instead of code.
- `tests/test_difference_map_artifact.py`: NEW — synthetic pair → success asserts `difference_map.png` exists/non-empty, `metrics.json` carries all 8 keys (in_sample_rmse, held_out_rmse, inlier_ratio, uniformity_score, cyclic_rmse, ssim_score, confidence_score, traffic_light_color), raster + matches JSON exist with ≥4 matches.
Tests:
- `python -m pytest tests/test_difference_map_artifact.py -v` → 1 passed (first run; failure-path honesty covered by existing `test_insufficient_matches_never_creates_fake_points`).
Evidence: command above.

## Task 7: ISRO official evaluator traffic-light summary
Status: PASS
Changes:
- `scripts/isro_official_evaluator.py`: pairwise records now emit `ssim_score`, `confidence_score`, `traffic_light_color`, `held_out_rmse`, `inlier_ratio`, `uniformity_score`, `cyclic_rmse` (all honest nulls when uncomputable); summary adds `average_ssim`, `average_confidence_score`, and `traffic_light_distribution` {GREEN,YELLOW,RED,UNKNOWN}; markdown adds two extra rows (SSIM, confidence, traffic-light tally). Never crashes on missing keys.
Tests:
- `python scripts/isro_official_evaluator.py --help` → 0 (argparse ok).
- Synthetic fixture: 2 failing pairs → pairwise keys present, distribution all zero; 1 success + 1 fail → YELLOW=1, average_ssim=0.1783, average_confidence=78, success/held-out handling correct, no fabricated metrics.
Evidence: commands where pairwise records and summary JSON were printed and inspected.

## Task 8: Full regression + FINAL_AGENT_REPORT.md
Status: PASS
Changes:
- `FINAL_AGENT_REPORT.md` created as required by the prompt.
Tests:
- Targeted: `tests/test_*epic* + submission_docs + difference_map + metrics + frontend_contract` → 30 passed; `test_registration_pipeline + phase2_pyramid + matcher_magsac` → 30 passed (1 skip); backend (`backend/test_api + test_security` incl. the two former flakes) → 62 passed (+ 2 previously failing now pass on a clean env); `test_ablation_smoke + ai_verifier_* + borrowed_features + candidate_generation` → 42 passed; `smoke` 9/9; `build` success. Live LRO network test correctly skips without connectivity (`tests/test_lro_ode_live_integration.py` → 1 skipped).
Evidence: batched `python -m pytest ... -q` runs above with stdout captured; frontend commands above.
