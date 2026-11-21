"""
BHAVIK Prediction Stabilization Module
Implements rolling temporal voting, confidence gating, hysteresis,
and cooldown management to prevent jitter and accidental duplicate insertions.
"""

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from backend.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_SMOOTHING_BUFFER_SIZE,
    DEFAULT_STABILITY_RATIO,
)


@dataclass
class SmootherResult:
    """State of the stabilization pipeline for the current frame."""
    hand_detected: bool
    predicted_letter: Optional[str]
    confidence: float
    is_stable: bool
    stability_progress: float  # Value between 0.0 and 1.0
    accepted_letter: Optional[str]  # Non-None only on the exact frame a letter is confirmed
    last_accepted_letter: Optional[str]
    status: str  # e.g., 'NO_HAND', 'STABILIZING', 'STABLE', 'ACCEPTED', 'COOLDOWN', 'HELD'


class PredictionSmoother:
    """
    Stabilizes noisy per-frame predictions using a sliding window majority vote,
    minimum stability duration, and cooldown prevention.
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_SMOOTHING_BUFFER_SIZE,
        stability_ratio: float = DEFAULT_STABILITY_RATIO,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self.buffer_size = max(3, buffer_size)
        self.stability_ratio = min(1.0, max(0.5, stability_ratio))
        self.confidence_threshold = min(1.0, max(0.1, confidence_threshold))
        self.cooldown_seconds = max(0.1, cooldown_seconds)

        self.buffer: Deque[Optional[str]] = collections.deque(maxlen=self.buffer_size)
        self.last_accepted_letter: Optional[str] = None
        self.last_accepted_time: float = 0.0
        self.held_letter: Optional[str] = None

    def configure(
        self,
        buffer_size: Optional[int] = None,
        stability_ratio: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        """Dynamically reconfigures smoothing parameters from user settings."""
        if buffer_size is not None and buffer_size > 0:
            self.buffer_size = buffer_size
            self.buffer = collections.deque(self.buffer, maxlen=self.buffer_size)
        if stability_ratio is not None:
            self.stability_ratio = min(1.0, max(0.5, stability_ratio))
        if confidence_threshold is not None:
            self.confidence_threshold = min(1.0, max(0.1, confidence_threshold))
        if cooldown_seconds is not None:
            self.cooldown_seconds = max(0.1, cooldown_seconds)

    def reset(self) -> None:
        """Resets the temporal buffer and gesture hold state."""
        self.buffer.clear()
        self.held_letter = None
        self.last_accepted_letter = None
        self.last_accepted_time = 0.0

    def process(
        self,
        predicted_letter: Optional[str],
        confidence: float,
        hand_detected: bool,
        current_time: Optional[float] = None,
    ) -> SmootherResult:
        """
        Processes a single frame's raw prediction and returns a smoothed result.
        """
        now = current_time if current_time is not None else time.time()

        # Case 1: No hand visible
        if not hand_detected:
            # When hand is removed, clear buffer and release held letter
            self.buffer.clear()
            self.held_letter = None
            return SmootherResult(
                hand_detected=False,
                predicted_letter=None,
                confidence=0.0,
                is_stable=False,
                stability_progress=0.0,
                accepted_letter=None,
                last_accepted_letter=self.last_accepted_letter,
                status="NO_HAND",
            )

        # Case 2: Low confidence prediction
        if predicted_letter is None or confidence < self.confidence_threshold:
            self.buffer.append(None)
            return SmootherResult(
                hand_detected=True,
                predicted_letter=predicted_letter,
                confidence=confidence,
                is_stable=False,
                stability_progress=0.0,
                accepted_letter=None,
                last_accepted_letter=self.last_accepted_letter,
                status="LOW_CONFIDENCE",
            )

        # Case 3: Valid confident prediction -> push to buffer
        self.buffer.append(predicted_letter)

        # Count occurrences of the current predicted letter in the buffer
        count = sum(1 for item in self.buffer if item == predicted_letter)
        stability_progress = min(1.0, count / (self.buffer_size * self.stability_ratio))
        is_stable = (count >= int(self.buffer_size * self.stability_ratio))

        accepted_letter: Optional[str] = None
        status = "STABILIZING"

        if is_stable:
            # Check cooldown and held-gesture state
            time_since_last = now - self.last_accepted_time
            in_cooldown = (predicted_letter == self.last_accepted_letter) and (time_since_last < self.cooldown_seconds)
            is_held = (predicted_letter == self.held_letter)

            if in_cooldown:
                status = "COOLDOWN"
            elif is_held:
                status = "HELD"
            else:
                # Accept gesture!
                accepted_letter = predicted_letter
                self.last_accepted_letter = predicted_letter
                self.last_accepted_time = now
                self.held_letter = predicted_letter
                status = "ACCEPTED"

        return SmootherResult(
            hand_detected=True,
            predicted_letter=predicted_letter,
            confidence=confidence,
            is_stable=is_stable,
            stability_progress=stability_progress,
            accepted_letter=accepted_letter,
            last_accepted_letter=self.last_accepted_letter,
            status=status,
        )
