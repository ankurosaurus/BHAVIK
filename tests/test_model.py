"""
Unit tests for BHAVIK Trained Machine Learning Model & Predictor
"""

import numpy as np
import pytest
from backend.config import METADATA_PATH, MODEL_PATH
from backend.src.predictor import ASLPredictor


@pytest.fixture
def predictor():
    pred = ASLPredictor(MODEL_PATH, METADATA_PATH)
    assert pred.is_loaded, "Model should be trained and saved before running model tests."
    return pred


def test_model_loading(predictor):
    assert predictor.is_loaded
    assert len(predictor.classes) == 26
    assert "A" in predictor.classes
    assert "J" in predictor.classes
    assert "Z" in predictor.classes
    assert "Y" in predictor.classes


def test_metadata_structure(predictor):
    meta = predictor.get_metadata()
    assert meta["model_type"] == "RandomForestClassifier"
    assert "metrics" in meta
    metrics = meta["metrics"]
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.85  # Real model achieves > 90%
    assert "per_class_report" in metrics


def test_prediction_output(predictor):
    # Dummy valid 63-feature vector
    dummy_feat = np.random.uniform(-1.0, 1.0, size=(63,)).astype(np.float32)
    dummy_feat[0:3] = 0.0  # Wrist at origin

    top_letter, confidence, prob_dict = predictor.predict(dummy_feat)
    assert isinstance(top_letter, str)
    assert 0.0 <= confidence <= 1.0
    assert len(prob_dict) == 26
    total_prob = sum(prob_dict.values())
    assert np.isclose(total_prob, 1.0, atol=1e-3)


def test_invalid_feature_handling(predictor):
    # Wrong shape
    letter, conf, probas = predictor.predict(np.zeros((10,)))
    assert letter is None
    assert conf == 0.0
    assert probas == {}

    # None
    letter, conf, probas = predictor.predict(None)
    assert letter is None
    assert conf == 0.0
