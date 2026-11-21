"""
BHAVIK Hand Detector Module
Encapsulates Google MediaPipe Hands for robust 21-landmark hand tracking.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import mediapipe as mp
import numpy as np

from backend.config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
)


class HandDetector:
    """Detects and extracts 21 3D hand landmarks from video frames using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.results: Optional[Any] = None

    def process_frame(self, image_bgr: np.ndarray) -> bool:
        """Processes a BGR image frame and returns True if at least one hand is detected."""
        if image_bgr is None or image_bgr.size == 0:
            self.results = None
            return False

        # OpenCV reads BGR; MediaPipe requires RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        self.results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        return bool(self.results and self.results.multi_hand_landmarks)

    def get_landmarks(
        self, hand_index: int = 0
    ) -> Optional[List[Dict[str, float]]]:
        """
        Returns the 21 normalized landmarks (x, y, z) for the specified hand.
        Coordinates are normalized to [0.0, 1.0] by MediaPipe.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        if hand_index >= len(self.results.multi_hand_landmarks):
            return None

        hand_landmarks = self.results.multi_hand_landmarks[hand_index]
        landmarks_list: List[Dict[str, float]] = []

        for lm in hand_landmarks.landmark:
            landmarks_list.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
            })

        return landmarks_list

    def get_landmarks_array(self, hand_index: int = 0) -> Optional[np.ndarray]:
        """Returns 21 landmarks as a numpy array of shape (21, 3)."""
        landmarks = self.get_landmarks(hand_index=hand_index)
        if landmarks is None:
            return None

        arr = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            arr[i, 0] = lm["x"]
            arr[i, 1] = lm["y"]
            arr[i, 2] = lm["z"]
        return arr

    def draw_landmarks(
        self, image_bgr: np.ndarray, hand_index: int = 0
    ) -> np.ndarray:
        """Annotates the BGR image with hand skeleton connections and landmarks."""
        annotated = image_bgr.copy()
        if not self.results or not self.results.multi_hand_landmarks:
            return annotated

        if hand_index < len(self.results.multi_hand_landmarks):
            hand_landmarks = self.results.multi_hand_landmarks[hand_index]
            self.mp_draw.draw_landmarks(
                annotated,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
