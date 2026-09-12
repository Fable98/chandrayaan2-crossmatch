import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger("ML_model.ai_verifier")

# Must match train_ai_verifier.FEATURE_NAMES and the saved bundle's feature order.
FEATURE_NAMES = [
    "confidence",
    "refinement_dx",
    "refinement_dy",
    "spatial_quality_score",
    "cfog_distance",
    "pc_energy_src",
    "pc_energy_tgt",
    "nn_ratio",
    "scale_diff",
]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ai_verifier_model.pkl"

_GIT_LFS_PREAMBLE = b"version https://git-lfs"


def looks_like_git_lfs_pointer(path: Path) -> bool:
    """True if *path* is an unpulled Git LFS pointer, not the real asset.

    joblib chokes on pointer text with a cryptic ``KeyError: 118``; sniffing
    the preamble lets the loader emit an actionable message instead.
    """
    try:
        if path.stat().st_size >= 4096:
            return False
        with open(path, "rb") as handle:
            return handle.read(len(_GIT_LFS_PREAMBLE) + 8).startswith(_GIT_LFS_PREAMBLE)
    except OSError:
        return False


class AIMatchVerifier:
    """
    Phase 4: AI Match Verification.

    Loads a hand-trained RandomForest bundle (``label_source=hand``) when
    present. Predicted inlier probabilities weight Phase 7 RANSAC. A hard
    pre-RANSAC veto remains opt-in (``experimental_stack=True``) because it
    rejected true low-MI OHRC↔TMC matches when NCC/MI confidence barely
    separated classes. RANSAC-labelled bundles are refused at load time
    (circular supervision). Missing/untrained path: documented non-ML
    percentile baseline for ``filter_matches`` only.
    """
    def __init__(self, model_path: Optional[str | Path] = None):
        self.model: Any = None
        self.is_trained = False
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.feature_names = list(FEATURE_NAMES)
        self._try_load_model(self.model_path)
        logger.info(f"AIMatchVerifier initialized (trained={self.is_trained}).")

    def _try_load_model(self, path: Path) -> bool:
        """Load a hand-trained joblib bundle; return True on success."""
        try:
            if not path.exists():
                logger.info(f"No trained model at {path}; using non-ML baseline (no model ships).")
                return False
            if looks_like_git_lfs_pointer(path):
                logger.warning(
                    "AI verifier model at %s is an unpulled Git LFS pointer "
                    "(joblib would fail with KeyError: 118). Run `git lfs pull "
                    "--include \"%s\"` (or re-clone with LFS) to materialize it. "
                    "Using non-ML baseline.",
                    path, path.name,
                )
                return False
            import joblib
            loaded = joblib.load(path)
            # train_ai_verifier saves {"model": clf, "feature_names": [...], ...}
            if isinstance(loaded, dict) and "model" in loaded:
                if loaded.get("label_source") not in ("hand", "hand_labelled"):
                    logger.warning("Model bundle label_source=%r is not hand-labelled; refusing to load "
                                   "(RANSAC-derived bundles are circular).",
                                   loaded.get("label_source"))
                    return False
                self.model = loaded["model"]
                self.feature_names = list(loaded.get("feature_names", FEATURE_NAMES))
            else:
                logger.warning("Model bundle at %s lacks provenance metadata; refusing to load.", path)
                return False
            # Sanity check: must expose predict_proba (i.e. actually fitted).
            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("loaded object has no predict_proba")
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
            except Exception as exc:
                raise AttributeError(f"loaded model is not fitted: {exc}")
            self.is_trained = True
            logger.info(f"Loaded trained AI verifier from {path}.")
            return True
        except Exception as e:
            logger.warning(f"Could not load AI verifier model from {path}: {e}. Using non-ML baseline.")
            self.model = None
            self.is_trained = False
            return False

    def extract_features(self, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for the AI model (order matches FEATURE_NAMES).

        Always materializes the 9-feature contract. If an older bundle was
        loaded (``self.feature_names`` is a prefix of FEATURE_NAMES), columns
        are selected to match ``n_features_in_`` so predict does not fail.
        """
        names = [n for n in list(self.feature_names or FEATURE_NAMES) if n in FEATURE_NAMES]
        if not names:
            names = list(FEATURE_NAMES)
        features = []
        for m in matches:
            conf = float(m.get("confidence", m.get("score", 0.0)))
            dx = float(m.get("refinement_dx", m.get("ref_dx", 0.0)))
            dy = float(m.get("refinement_dy", m.get("ref_dy", 0.0)))
            spatial_raw = m.get("spatial_quality_score", m.get("spatial_score", None))
            if spatial_raw is None:
                spatial = 1.0 if m.get("is_refined", False) else 0.5
            else:
                try:
                    spatial = float(spatial_raw)
                except (TypeError, ValueError):
                    spatial = 0.5
            vec = {
                "confidence": conf,
                "refinement_dx": dx,
                "refinement_dy": dy,
                "spatial_quality_score": spatial,
                "cfog_distance": float(m.get("cfog_distance", 0.0) or 0.0),
                "pc_energy_src": float(m.get("pc_energy_src", 0.0) or 0.0),
                "pc_energy_tgt": float(m.get("pc_energy_tgt", 0.0) or 0.0),
                "nn_ratio": float(m.get("nn_ratio", 1.0) if m.get("nn_ratio") is not None else 1.0),
                "scale_diff": float(m.get("scale_diff", 0.0) or 0.0),
            }
            features.append([vec[n] for n in names])
        return np.array(features, dtype=np.float64)

    def predict_confidence(self, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Returns probability each match is a true inlier."""
        if len(matches) == 0:
            return np.array([])

        # Extract coarse confidence scores
        scores = np.array([
            float(m.get("confidence", m.get("score", 0.5))) for m in matches
        ], dtype=np.float32)

        if not self.is_trained:
            # NON-ML BASELINE (not a learned model): keep matches at or above
            # the 25th percentile of the current batch. Documented as a weak
            # heuristic, NOT an AI verdict.
            threshold = float(np.percentile(scores, 25)) if len(scores) > 4 else 0.3
            # Return 1.0 for pass, 0.0 for fail (simulating probability)
            return (scores >= threshold).astype(np.float32)

        features = self.extract_features(matches)
        try:
            proba = self.model.predict_proba(features)
            # Handle single-class edge: predict_proba may return 1 column.
            if proba.shape[1] == 1:
                cls = list(getattr(self.model, "classes_", [0]))
                if cls and int(cls[0]) == 1:
                    return np.ones(len(matches), dtype=np.float32)
                return np.zeros(len(matches), dtype=np.float32)
            # Column for class 1 (inlier); fall back to last column.
            classes = list(getattr(self.model, "classes_", [0, 1]))
            col = classes.index(1) if 1 in classes else -1
            return np.asarray(proba[:, col], dtype=np.float32)
        except Exception as e:
            logger.warning(f"AI Verifier prediction failed: {e}")
            return (scores >= 0.3).astype(np.float32)

    def filter_matches(self, matches: List[Dict[str, Any]], threshold: float = 0.5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Outlier rejection via the loaded model, or the non-ML baseline."""
        confidences = self.predict_confidence(matches)
        kept = []
        rejected = []
        for m, conf in zip(matches, confidences):
            if conf >= threshold:
                kept.append(m)
            else:
                rejected.append(m)
        logger.info(f"AI Verifier: Kept {len(kept)} matches, rejected {len(rejected)}")
        return kept, rejected
