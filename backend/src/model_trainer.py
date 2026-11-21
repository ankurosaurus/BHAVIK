"""
BHAVIK Model Trainer Module
Trains, evaluates, and persists the Scikit-Learn Random Forest Classifier
for 24-class ASL alphabet fingerspelling recognition.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from backend.config import (
    DATASET_CSV_PATH,
    METADATA_PATH,
    MODEL_PATH,
    SUPPORTED_CLASSES,
)
from backend.src.feature_extractor import FEATURE_NAMES, NUM_FEATURES


class ModelTrainer:
    """Manages training, cross-validation, and performance evaluation for ASL gestures."""

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_CSV_PATH
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: Optional[np.ndarray] = None
        self.evaluation_results: Dict[str, Any] = {}

    def load_and_preprocess(
        self, test_size: float = 0.20, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads CSV, validates 63 numerical features and valid classes,
        and creates a stratified train/test split.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # Validate mandatory columns
        if "label" not in df.columns:
            raise ValueError("Dataset is missing 'label' column.")

        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset is missing {len(missing_features)} feature columns.")

        # Filter strictly to supported classes
        df = df[df["label"].isin(SUPPORTED_CLASSES)].copy()

        # Drop any row with NaN or infinite values
        feature_subset = df[FEATURE_NAMES]
        valid_mask = feature_subset.notna().all(axis=1) & np.isfinite(feature_subset).all(axis=1)
        df = df[valid_mask].copy()

        if len(df) < 50:
            raise ValueError(f"Insufficient training samples ({len(df)} rows). Need at least 50.")

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["label"].values.astype(str)

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 150,
        max_depth: int = 20,
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> RandomForestClassifier:
        """Trains a Scikit-Learn RandomForestClassifier."""
        print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_duration = time.time() - start_time
        self.classes_ = self.model.classes_

        print(f"Model trained in {training_duration:.2f} seconds. Classes: {len(self.classes_)}")
        return self.model

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates authentic evaluation metrics on unseen test data."""
        if self.model is None:
            raise RuntimeError("Cannot evaluate an untrained model. Call train() first.")

        # Latency benchmark (measure average ms per single prediction)
        latency_samples = min(100, len(X_test))
        t_start = time.perf_counter()
        for i in range(latency_samples):
            _ = self.model.predict_proba([X_test[i]])
        t_elapsed = time.perf_counter() - t_start
        avg_latency_ms = (t_elapsed / latency_samples) * 1000.0

        # Batch prediction
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        c_matrix = confusion_matrix(y_test, y_pred, labels=self.classes_).tolist()

        self.evaluation_results = {
            "accuracy": acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "average_latency_ms": avg_latency_ms,
            "test_samples": int(len(y_test)),
            "per_class_report": report,
            "confusion_matrix": c_matrix,
            "classes": [str(c) for c in self.classes_],
        }

        print("=" * 60)
        print("BHAVIK Model Evaluation Summary")
        print("=" * 60)
        print(f"Test Accuracy:       {acc * 100:.2f}%")
        print(f"Macro Precision:     {precision_macro * 100:.2f}%")
        print(f"Macro Recall:        {recall_macro * 100:.2f}%")
        print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
        print(f"Prediction Latency:  {avg_latency_ms:.3f} ms / frame")
        print("=" * 60)

        return self.evaluation_results

    def save(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Saves trained model and metadata."""
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        m_path = Path(model_path) if model_path else MODEL_PATH
        meta_path = Path(metadata_path) if metadata_path else METADATA_PATH

        m_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, m_path, compress=3)
        print(f"Model saved to {m_path}")

        metadata = {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "classes": [str(c) for c in self.classes_],
            "num_features": NUM_FEATURES,
            "feature_names": FEATURE_NAMES,
            "metrics": self.evaluation_results,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {meta_path}")

        return m_path, meta_path
