import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("ML_model.ai_verifier")

class AIMatchVerifier:
    """
    Phase 4: AI Match Verification
    Purpose: Use Supervised Machine Learning to filter out false-positive matches.
    Method: RandomForestClassifier trained on match confidence features.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.is_trained = False
        logger.info("AIMatchVerifier initialized.")

    def extract_features(self, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for the AI model."""
        features = []
        for m in matches:
            features.append([
                float(m.get("confidence", m.get("score", 0.0))),           # Coarse matching score
                float(m.get("refinement_dx", 0.0)),   # Sub-pixel shift X
                float(m.get("refinement_dy", 0.0)),   # Sub-pixel shift Y
                1.0 if m.get("is_refined", False) else 0.0  # Was it refined?
            ])
        return np.array(features)

    def predict_confidence(self, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Returns probability each match is a true inlier."""
        if len(matches) == 0:
            return np.array([])
        
        # Extract coarse confidence scores
        scores = np.array([
            float(m.get("confidence", m.get("score", 0.5))) for m in matches
        ], dtype=np.float32)

        if not self.is_trained:
            # HEURISTIC FALLBACK: 
            # If no model is trained, act as a strict statistical gate.
            # Reject matches that fall below the 25th percentile of the current batch.
            threshold = float(np.percentile(scores, 25)) if len(scores) > 4 else 0.3
            # Return 1.0 for pass, 0.0 for fail (simulating probability)
            return (scores >= threshold).astype(np.float32)
        
        features = self.extract_features(matches)
        try:
            return self.model.predict_proba(features)[:, 1]
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
