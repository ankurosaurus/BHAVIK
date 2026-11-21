"""
Unit tests for BHAVIK Prediction Smoother
"""

import pytest
from backend.src.prediction_smoother import PredictionSmoother


@pytest.fixture
def smoother():
    return PredictionSmoother(
        buffer_size=5,
        stability_ratio=0.8,  # requires 4 out of 5 frames
        confidence_threshold=0.70,
        cooldown_seconds=1.0,
    )


def test_no_hand_state(smoother):
    res = smoother.process(None, 0.0, hand_detected=False)
    assert not res.hand_detected
    assert res.accepted_letter is None
    assert res.status == "NO_HAND"


def test_low_confidence_rejection(smoother):
    res = smoother.process("A", 0.50, hand_detected=True)
    assert res.hand_detected
    assert res.accepted_letter is None
    assert res.status == "LOW_CONFIDENCE"


def test_stabilization_and_acceptance(smoother):
    # Feed 3 'B's -> not stable yet (needs 4 out of 5)
    for _ in range(3):
        res = smoother.process("B", 0.90, hand_detected=True, current_time=100.0)
        assert res.accepted_letter is None
        assert res.status == "STABILIZING"

    # 4th 'B' -> meets threshold! Must accept!
    res = smoother.process("B", 0.90, hand_detected=True, current_time=100.1)
    assert res.is_stable
    assert res.accepted_letter == "B"
    assert res.status == "ACCEPTED"


def test_held_gesture_prevention(smoother):
    # Stabilize and accept 'A'
    for _ in range(4):
        _ = smoother.process("A", 0.95, hand_detected=True, current_time=200.0)

    # 5th frame with 'A' still held -> must not accept duplicate!
    res = smoother.process("A", 0.95, hand_detected=True, current_time=200.2)
    assert res.accepted_letter is None
    assert res.status in ["HELD", "COOLDOWN"]


def test_distinct_gesture_transition(smoother):
    # Accept 'A'
    for _ in range(4):
        _ = smoother.process("A", 0.95, hand_detected=True, current_time=300.0)

    # Transition to 'B'
    for _ in range(4):
        res = smoother.process("B", 0.92, hand_detected=True, current_time=300.5)

    # Should accept 'B'
    assert res.accepted_letter == "B"
    assert res.status == "ACCEPTED"


def test_cooldown_expiry_after_hand_reappearance(smoother):
    # Accept 'A'
    for _ in range(4):
        _ = smoother.process("A", 0.95, hand_detected=True, current_time=400.0)

    # Hand removed
    smoother.process(None, 0.0, hand_detected=False, current_time=400.5)

    # Re-appear with 'A' after cooldown period (> 1.0s)
    for _ in range(3):
        smoother.process("A", 0.95, hand_detected=True, current_time=401.5)

    res = smoother.process("A", 0.95, hand_detected=True, current_time=401.6)
    assert res.accepted_letter == "A"
    assert res.status == "ACCEPTED"
