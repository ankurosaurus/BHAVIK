"""
BHAVIK Single-Command Model Training Script
Trains, evaluates, and exports the Random Forest Classifier.
Usage:
    python backend/scripts/train_model.py [--estimators 150] [--depth 20]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import DATASET_CSV_PATH, METADATA_PATH, MODEL_PATH
from backend.src.model_trainer import ModelTrainer
from backend.src.predictor import ASLPredictor


def run_training(n_estimators: int = 150, max_depth: int = 20):
    print("=" * 60)
    print("BHAVIK: Initiating ASL Fingerspelling Model Training")
    print("=" * 60)

    trainer = ModelTrainer(dataset_path=DATASET_CSV_PATH)
    try:
        X_train, X_test, y_train, y_test = trainer.load_and_preprocess(test_size=0.20, random_state=42)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_CSV_PATH}. Running benchmark dataset generator first...")
        from backend.scripts.generate_asl_dataset import generate_benchmark_dataset
        generate_benchmark_dataset()
        X_train, X_test, y_train, y_test = trainer.load_and_preprocess(test_size=0.20, random_state=42)

    print(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} validation samples.")
    print(f"Features: {X_train.shape[1]} invariant landmark coordinates.")

    trainer.train(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )

    trainer.evaluate(X_test, y_test)
    m_path, meta_path = trainer.save(MODEL_PATH, METADATA_PATH)

    # Verification step
    print("\nVerifying model loading and inference via ASLPredictor...")
    predictor = ASLPredictor(m_path, meta_path)
    assert predictor.is_loaded, "Verification failed: Predictor could not load the saved model!"

    # Test single prediction
    sample_feat = X_test[0]
    pred_letter, conf, _ = predictor.predict(sample_feat)
    print(f"Verification sample prediction: Expected '{y_test[0]}', Got '{pred_letter}' (Confidence: {conf*100:.1f}%)")
    print("\nModel training, evaluation, and verification completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BHAVIK ASL Fingerspelling Classifier")
    parser.add_argument("--estimators", type=int, default=150, help="Number of trees in Random Forest")
    parser.add_argument("--depth", type=int, default=20, help="Maximum tree depth")
    args = parser.parse_args()
    run_training(n_estimators=args.estimators, max_depth=args.depth)
