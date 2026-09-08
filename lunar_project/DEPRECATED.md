# DEPRECATION NOTICE: `lunar_project/`

As part of the SIH26166 architectural consolidation, the entire `lunar_project/` tree has been **formally deprecated**.

---

## What was deprecated?

1. **Unused Vite Frontend Prototype (`lunar_project/frontend/`)**:
   - The official production web application is located in `lunar-frontend/` (Next.js 14 App Router, deployed live on Vercel).
2. **Duplicate Geospatial & Machine Learning Scripts (`lunar_project/src/`)**:
   - Georeferencing, ingestion, and preprocessing have been consolidated into `data_preprocessing_pipeline/`.
   - Core photogrammetric cross-matching, Phase Congruency, CFOG, and sub-pixel phase correlation have been consolidated into `ML_model/matcher_cfog.py`.
   - Triplet closed-loop cycle consistency evaluation has been consolidated into `data_preprocessing_pipeline/triplet_evaluator.py`.

---

## Active Code Locations

| Functionality | Old Location (Deprecated) | Active Location |
| :--- | :--- | :--- |
| **PDS4 / Raw Ingestion** | `lunar_project/src/data/` | `data_preprocessing_pipeline/pipeline.py` |
| **DEM Orthorectification** | N/A | `data_preprocessing_pipeline/pipeline.py` |
| **Phase Congruency & CFOG Matching** | `lunar_project/src/ml/` | `ML_model/matcher_cfog.py` |
| **Triplet Cycle Consistency** | `lunar_project/src/ml/evaluation/` | `data_preprocessing_pipeline/triplet_evaluator.py` |
| **FastAPI Backend & API** | `lunar_project/src/backend/` | `backend/main.py` |
| **Production Web UI** | `lunar_project/frontend/` | `lunar-frontend/` |

Do not add new features or depend on files inside `lunar_project/`. All active development targets `data_preprocessing_pipeline/`, `ML_model/`, `backend/`, and `lunar-frontend/`.
