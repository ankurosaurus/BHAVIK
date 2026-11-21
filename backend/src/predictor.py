"""
BHAVIK Real-Time ASL Predictor Module
Executes low-latency inference on normalized landmark features.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from backend.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    METADATA_PATH,
    MODEL_PATH,
    SUPPORTED_CLASSES,
)
from backend.src.feature_extractor import NUM_FEATURES


class ASLPredictor:
    """Loads trained model and provides real-time confidence-rated inference."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.confidence_threshold = confidence_threshold

        self.model: Optional[Any] = None
        self.classes: List[str] = list(SUPPORTED_CLASSES)
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self.load_model()

    def load_model(self) -> bool:
        """Loads model binary from disk if available."""
        if not self.model_path.exists():
            self.is_loaded = False
            return False

        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "classes_"):
                self.classes = [str(c) for c in self.model.classes_]

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            self.is_loaded = False
            return False

    def predict(
        self, features: np.ndarray
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Runs inference on a (63,) normalized feature vector.
        Returns:
            (top_letter, confidence_score, class_probabilities_dict)
            If confidence is below threshold or model is not loaded, top_letter may be None.
        """
        if not self.is_loaded or self.model is None:
            return None, 0.0, {}

        if features is None or features.shape != (NUM_FEATURES,):
            return None, 0.0, {}

        try:
            # Predict probabilities
            probas = self.model.predict_proba([features])[0]
            max_idx = int(np.argmax(probas))
            top_confidence = float(probas[max_idx])
            top_letter = str(self.classes[max_idx])

            prob_dict = {
                str(cls_name): float(probas[i]) for i, cls_name in enumerate(self.classes)
            }

            return top_letter, top_confidence, prob_dict
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, {}

    def get_metadata(self) -> Dict[str, Any]:
        """Returns loaded model metadata."""
        return dict(self.metadata)
