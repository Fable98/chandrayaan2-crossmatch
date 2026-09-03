# DEPRECATION NOTICE: `processed_user/`

**Status**: Deprecated mirror directory.

All canonical preprocessed dataset triplets and match files are maintained in:
- `data_preprocessing_pipeline/processed_triplets/`
- `data_preprocessing_pipeline/matches/`

The match files in `processed_user/matches/` are synchronized with `data_preprocessing_pipeline/matches/` for backwards compatibility with legacy tooling, but all active pipelines and APIs read directly from `data_preprocessing_pipeline/`.
