import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger("ML_model.ai_verifier")

# Must match train_ai_verifier.FEATURE_NAMES and the saved bundle's feature order.
FEATURE_NAMES = ["confidence", "refinement_dx", "refinement_dy", "spatial_quality_score"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ai_verifier_model.pkl"


class AIMatchVerifier:
    """
    Phase 4: AI Match Verification
    Purpose: Use Supervised Machine Learning to filter out false-positive matches.
    Method: RandomForestClassifier trained on match confidence features.

    If a trained ``ai_verifier_model.pkl`` (written by train_ai_verifier.py)
    exists next to this file it is loaded in __init__ and used for
    predict_proba. Otherwise the verifier falls back to the legacy
    percentile heuristic.
    """
    def __init__(self, model_path: Optional[str | Path] = None):
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=None, class_weight="balanced"
        )
        self.is_trained = False
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.feature_names = list(FEATURE_NAMES)
        self._try_load_model(self.model_path)
        logger.info(f"AIMatchVerifier initialized (trained={self.is_trained}).")

    def _try_load_model(self, path: Path) -> bool:
        """Load a joblib bundle (or bare classifier); return True on success."""
        try:
            if not path.exists():
                logger.info(f"No trained model at {path}; using heuristic fallback.")
                return False
            import joblib
            loaded = joblib.load(path)
            # train_ai_verifier saves {"model": clf, "feature_names": [...], ...}
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded["model"]
                self.feature_names = list(loaded.get("feature_names", FEATURE_NAMES))
            else:
                self.model = loaded
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
            logger.warning(f"Could not load AI verifier model from {path}: {e}. Using heuristic fallback.")
            self.is_trained = False
            return False

    def extract_features(self, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for the AI model (order matches FEATURE_NAMES)."""
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
            features.append([conf, dx, dy, spatial])
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
            # HEURISTIC FALLBACK (only when no .pkl model file exists):
            # Reject matches that fall below the 25th percentile of the current batch.
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
        """AI-powered outlier rejection."""
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
