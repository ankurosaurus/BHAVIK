"""
BHAVIK Real Landmark Dataset Importer
Transforms real human MediaPipe hand landmark captures across all 26 ASL alphabet classes
into translation-invariant, scale-invariant, dual-handed normalized dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from backend.config import DATASET_CSV_PATH, SUPPORTED_CLASSES
from backend.src.feature_extractor import FeatureExtractor, FEATURE_NAMES

def import_and_normalize_real_dataset(
    source_csv: Path,
    target_csv: Path = DATASET_CSV_PATH,
    max_samples_per_class: int = 1500,
):
    print(f"Reading real MediaPipe landmark dataset from {source_csv}...")
    df_raw = pd.read_csv(source_csv)

    # Filter to 26 ASL alphabet letters
    df_raw = df_raw[df_raw["label"].isin(SUPPORTED_CLASSES)].copy()
    print(f"Loaded {len(df_raw)} genuine hand landmark samples across 26 classes.")

    extractor = FeatureExtractor()
    raw_feature_cols = [c for c in df_raw.columns if c != "label"]

    processed_rows = []
    class_counts = {c: 0 for c in SUPPORTED_CLASSES}

    for idx, row in df_raw.iterrows():
        label = str(row["label"])
        if class_counts[label] >= max_samples_per_class:
            continue

        coords_flat = row[raw_feature_cols].values.astype(np.float32)
        if len(coords_flat) != 63:
            continue

        coords_21x3 = coords_flat.reshape((21, 3))

        # 1. Standard (Right hand orientation)
        norm_feat = extractor.normalize_landmarks(coords_21x3, is_left_hand=False)
        row_dict = {
            "label": label,
            "session_id": f"real_signer_{idx // 1000}",
            "is_left_hand": 0,
        }
        for name, val in zip(FEATURE_NAMES, norm_feat):
            row_dict[name] = float(val)
        processed_rows.append(row_dict)

        # 2. Add Left hand mirrored variant for 100% dual-handed accessibility
        norm_feat_left = extractor.normalize_landmarks(coords_21x3, is_left_hand=True)
        row_dict_left = {
            "label": label,
            "session_id": f"real_signer_left_{idx // 1000}",
            "is_left_hand": 1,
        }
        for name, val in zip(FEATURE_NAMES, norm_feat_left):
            row_dict_left[name] = float(val)
        processed_rows.append(row_dict_left)

        class_counts[label] += 1

    df_out = pd.DataFrame(processed_rows)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(target_csv, index=False)
    print(f"Successfully processed and normalized {len(df_out)} dual-handed real samples!")
    print(f"Saved dataset to {target_csv}")
    return df_out

if __name__ == "__main__":
    source = Path("temp_khaled/asl_landmark_features.csv")
    import_and_normalize_real_dataset(source)
