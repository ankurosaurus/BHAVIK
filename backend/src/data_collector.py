"""
BHAVIK Data Collector Module
Manages labelled landmark datasets for ASL fingerspelling recognition.
Ensures persistent storage, deduplication, incremental updates, and validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from backend.config import DATASET_CSV_PATH, PROCESSED_DATA_DIR, SUPPORTED_CLASSES
from backend.src.feature_extractor import FeatureExtractor, FEATURE_NAMES, NUM_FEATURES


class DataCollector:
    """Manages collection, persistence, and loading of ASL hand landmark datasets."""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else DATASET_CSV_PATH
        self.extractor = FeatureExtractor()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_dataset_file()

    def _ensure_dataset_file(self) -> None:
        """Initializes the CSV file with appropriate header columns if it doesn't exist."""
        if not self.csv_path.exists():
            headers = ["label", "session_id", "is_left_hand"] + FEATURE_NAMES
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_path, index=False)

    def add_sample(
        self,
        label: str,
        landmarks: Union[np.ndarray, List[Dict[str, float]]],
        session_id: str = "default_session",
        is_left_hand: bool = False,
    ) -> bool:
        """
        Extracts invariant features from 21 landmarks and appends the sample to the dataset.
        Returns True if successful, False if invalid hand landmarks.
        """
        label = label.upper().strip()
        if label not in SUPPORTED_CLASSES:
            return False

        if isinstance(landmarks, list):
            features = self.extractor.extract_from_dict_list(landmarks, is_left_hand=is_left_hand)
        elif isinstance(landmarks, np.ndarray):
            features = self.extractor.extract_from_array(landmarks, is_left_hand=is_left_hand)
        else:
            return False

        if features is None or len(features) != NUM_FEATURES:
            return False

        # Construct row
        row_dict = {
            "label": label,
            "session_id": str(session_id),
            "is_left_hand": int(is_left_hand),
        }
        for name, val in zip(FEATURE_NAMES, features):
            row_dict[name] = float(val)

        row_df = pd.DataFrame([row_dict])
        row_df.to_csv(self.csv_path, mode="a", header=not self.csv_path.exists(), index=False)
        return True

    def get_sample_counts(self) -> Dict[str, int]:
        """Returns the count of samples collected for each supported class."""
        counts = {c: 0 for c in SUPPORTED_CLASSES}
        if not self.csv_path.exists():
            return counts

        try:
            df = pd.read_csv(self.csv_path, usecols=["label"])
            val_counts = df["label"].value_counts().to_dict()
            for c in counts:
                counts[c] = int(val_counts.get(c, 0))
        except Exception:
            pass

        return counts

    def get_total_sample_count(self) -> int:
        """Returns total valid samples in dataset."""
        counts = self.get_sample_counts()
        return sum(counts.values())

    def load_dataset(self) -> pd.DataFrame:
        """Loads and returns the complete dataset as a pandas DataFrame."""
        if not self.csv_path.exists():
            return pd.DataFrame(columns=["label", "session_id", "is_left_hand"] + FEATURE_NAMES)
        return pd.read_csv(self.csv_path)

    def clear_dataset(self) -> None:
        """Clears dataset (resets CSV headers)."""
        headers = ["label", "session_id", "is_left_hand"] + FEATURE_NAMES
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.csv_path, index=False)
