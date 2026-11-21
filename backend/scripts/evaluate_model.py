"""
BHAVIK Detailed Model Evaluation Script
Computes comprehensive per-class precision, recall, F1, latency, and comparison with baseline SVM.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from backend.config import DATASET_CSV_PATH, MODEL_PATH, METADATA_PATH
from backend.src.model_trainer import ModelTrainer
from backend.src.predictor import ASLPredictor


def run_evaluation():
    print("=" * 70)
    print("BHAVIK Comprehensive Model Evaluation & Baseline Comparison")
    print("=" * 70)

    trainer = ModelTrainer(dataset_path=DATASET_CSV_PATH)
    X_train, X_test, y_train, y_test = trainer.load_and_preprocess(test_size=0.20, random_state=42)

    # Load existing trained Random Forest
    predictor = ASLPredictor(MODEL_PATH, METADATA_PATH)
    if not predictor.is_loaded:
        print("Trained model not found. Training Random Forest model first...")
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)
        trainer.save(MODEL_PATH, METADATA_PATH)
        predictor = ASLPredictor(MODEL_PATH, METADATA_PATH)

    metadata = predictor.get_metadata()
    metrics = metadata.get("metrics", {})

    print(f"\n[RANDOM FOREST CLASSIFIER (Production Model)]")
    print(f"Overall Accuracy:       {metrics.get('accuracy', 0)*100:.2f}%")
    print(f"Macro F1-Score:         {metrics.get('f1_macro', 0)*100:.2f}%")
    print(f"Inference Latency:      {metrics.get('average_latency_ms', 0):.3f} ms / frame")

    # Train baseline SVM for comparative study (as requested in Section 3)
    print("\nTraining Baseline Support Vector Machine (RBF Kernel) for comparison...")
    t0 = time.time()
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_train_time = time.time() - t0

    t_start = time.perf_counter()
    y_pred_svm = svm.predict(X_test)
    svm_latency_ms = ((time.perf_counter() - t_start) / len(X_test)) * 1000.0
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average="macro")

    print(f"\n[SUPPORT VECTOR MACHINE (Baseline Comparison)]")
    print(f"Training Time:          {svm_train_time:.2f} s")
    print(f"Overall Accuracy:       {svm_acc*100:.2f}%")
    print(f"Macro F1-Score:         {svm_f1*100:.2f}%")
    print(f"Inference Latency:      {svm_latency_ms:.3f} ms / frame")

    print("\n" + "=" * 70)
    print("Per-Class Classification Report (Random Forest):")
    print("=" * 70)
    print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
    print("-" * 54)
    per_class = metrics.get("per_class_report", {})
    for cls_name, vals in per_class.items():
        if isinstance(vals, dict):
            prec = vals.get("precision", 0) * 100
            rec = vals.get("recall", 0) * 100
            f1 = vals.get("f1-score", 0) * 100
            supp = vals.get("support", 0)
            print(f"{cls_name:<8}{prec:<12.1f}{rec:<12.1f}{f1:<12.1f}{supp:<10}")


if __name__ == "__main__":
    run_evaluation()
