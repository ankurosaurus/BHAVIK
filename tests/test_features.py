"""
Tests for BHAVIK Feature Extractor
"""

import numpy as np
import pytest
from backend.src.feature_extractor import FeatureExtractor, NUM_FEATURES, FEATURE_NAMES


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def sample_hand_landmarks():
    """Generates synthetic 21 3D points resembling a hand layout."""
    np.random.seed(42)
    # Wrist at (0.5, 0.8, 0.0)
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [0.5, 0.8, 0.0]
    # Fingers spread upward
    for i in range(1, 21):
        points[i] = [0.5 + 0.01 * (i % 5 - 2), 0.8 - 0.03 * (i // 4 + 1), 0.005 * i]
    return points


def test_feature_count_and_names(extractor):
    assert len(FEATURE_NAMES) == 63
    assert len(extractor.get_feature_names()) == 63
    assert extractor.get_feature_names()[0] == "x_0"
    assert extractor.get_feature_names()[1] == "y_0"
    assert extractor.get_feature_names()[2] == "z_0"
    assert extractor.get_feature_names()[-1] == "z_20"


def test_output_shape(extractor, sample_hand_landmarks):
    features = extractor.extract_from_array(sample_hand_landmarks)
    assert features is not None
    assert features.shape == (63,)
    assert features.dtype == np.float32


def test_wrist_at_origin(extractor, sample_hand_landmarks):
    features = extractor.extract_from_array(sample_hand_landmarks)
    # The first 3 elements (wrist normalized relative to itself) must be 0
    assert np.allclose(features[0:3], [0.0, 0.0, 0.0], atol=1e-6)


def test_scale_normalization(extractor, sample_hand_landmarks):
    features = extractor.extract_from_array(sample_hand_landmarks)
    # Reshape back to (21, 3) to test max distance
    coords = features.reshape((21, 3))
    distances = np.linalg.norm(coords, axis=1)
    # Maximum distance must be exactly 1.0 (within floating-point precision)
    assert np.isclose(np.max(distances), 1.0, atol=1e-5)


def test_translation_invariance(extractor, sample_hand_landmarks):
    # Shift hand in space by (dx, dy, dz)
    shifted_landmarks = sample_hand_landmarks + np.array([0.35, -0.42, 0.15], dtype=np.float32)
    feat_orig = extractor.extract_from_array(sample_hand_landmarks)
    feat_shifted = extractor.extract_from_array(shifted_landmarks)
    assert np.allclose(feat_orig, feat_shifted, atol=1e-5)


def test_scale_invariance(extractor, sample_hand_landmarks):
    # Scale hand by 2.5x
    wrist = sample_hand_landmarks[0]
    scaled_landmarks = (sample_hand_landmarks - wrist) * 2.5 + wrist
    feat_orig = extractor.extract_from_array(sample_hand_landmarks)
    feat_scaled = extractor.extract_from_array(scaled_landmarks)
    assert np.allclose(feat_orig, feat_scaled, atol=1e-5)


def test_dict_list_input(extractor, sample_hand_landmarks):
    dict_list = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in sample_hand_landmarks]
    features = extractor.extract_from_dict_list(dict_list)
    assert features is not None
    assert features.shape == (63,)


def test_invalid_inputs(extractor):
    assert extractor.extract_from_array(None) is None
    assert extractor.extract_from_array(np.zeros((10, 3))) is None
    assert extractor.extract_from_dict_list([]) is None
    assert extractor.extract_from_dict_list([{"x": 0.1, "y": 0.2}] * 20) is None  # Only 20 items
