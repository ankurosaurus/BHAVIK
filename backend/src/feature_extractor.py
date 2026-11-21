"""
BHAVIK Feature Extraction Module
Transforms raw 21 3D MediaPipe landmarks into a 63-dimensional
translation-invariant and scale-invariant feature vector.
"""

from typing import Dict, List, Optional, Union
import numpy as np


FEATURE_NAMES = [
    f"{axis}_{i}" for i in range(21) for axis in ("x", "y", "z")
]
NUM_FEATURES = 63  # 21 landmarks * 3 coordinates


class FeatureExtractor:
    """
    Extracts normalized, translation- and scale-invariant features from 21 hand landmarks.
    Ensures absolute consistency between training and real-time live inference.
    """

    def __init__(self, flip_left_hand: bool = False) -> None:
        self.flip_left_hand = flip_left_hand

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns the list of 63 feature names."""
        return list(FEATURE_NAMES)

    def extract_from_dict_list(
        self,
        landmarks: List[Dict[str, float]],
        is_left_hand: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Converts a list of 21 landmark dictionaries [{'x': .., 'y': .., 'z': ..}]
        into a normalized 63-element feature vector.
        """
        if not landmarks or len(landmarks) != 21:
            return None

        try:
            arr = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(landmarks):
                arr[i, 0] = float(lm["x"])
                arr[i, 1] = float(lm["y"])
                arr[i, 2] = float(lm.get("z", 0.0))
            return self.normalize_landmarks(arr, is_left_hand=is_left_hand)
        except (KeyError, TypeError, ValueError):
            return None

    def extract_from_array(
        self,
        landmarks: np.ndarray,
        is_left_hand: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extracts features from a numpy array of shape (21, 3) or flat (63,).
        """
        if landmarks is None:
            return None

        if landmarks.ndim == 1 and landmarks.shape[0] == NUM_FEATURES:
            arr = landmarks.reshape((21, 3)).astype(np.float32)
        elif landmarks.ndim == 2 and landmarks.shape == (21, 3):
            arr = landmarks.astype(np.float32)
        else:
            return None

        return self.normalize_landmarks(arr, is_left_hand=is_left_hand)

    def normalize_landmarks(
        self,
        landmarks_21x3: np.ndarray,
        is_left_hand: bool = False,
    ) -> np.ndarray:
        """
        Normalizes 21 3D landmarks:
        1. Translation Invariance: Shift all landmarks relative to the wrist (landmark 0).
        2. Scale Invariance: Divide by the maximum Euclidean distance from the wrist.
        3. Handedness Consistency: Optional mirroring along the X-axis for left hands.
        4. Flatten into 63-element 1D array.
        """
        if landmarks_21x3.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {landmarks_21x3.shape}")

        # Step 1: Wrist-centric translation invariance
        wrist = landmarks_21x3[0].copy()
        rel_coords = landmarks_21x3 - wrist  # wrist becomes [0, 0, 0]

        # Step 2: Scale invariance (palm/hand span normalization)
        distances = np.linalg.norm(rel_coords, axis=1)
        max_dist = float(np.max(distances))

        # Epsilon guard against division by zero
        if max_dist < 1e-6:
            scale = 1.0
        else:
            scale = max_dist

        norm_coords = rel_coords / scale

        # Step 3: Left/right hand consistency (mirroring X if enabled for left hand)
        if (self.flip_left_hand or is_left_hand) and is_left_hand:
            norm_coords[:, 0] = -norm_coords[:, 0]

        # Step 4: Flatten to (63,)
        features = norm_coords.flatten().astype(np.float32)
        return features
