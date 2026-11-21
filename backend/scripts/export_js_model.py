"""
Exports trained Scikit-Learn Random Forest Classifier to optimized JavaScript
for in-browser client-side inference on GitHub Pages without requiring a backend server.
"""

import json
from pathlib import Path
import joblib
import numpy as np

def export_model_to_js(model_path: Path, output_js_path: Path, num_trees: int = 60):
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    
    classes = [str(c) for c in model.classes_]
    trees_data = []

    selected_estimators = model.estimators_[:num_trees]
    print(f"Exporting top {len(selected_estimators)} decision trees...")

    for est in selected_estimators:
        t = est.tree_
        # Find dominant class index in each node
        leaf_classes = np.argmax(t.value[:, 0, :], axis=1).astype(int).tolist()
        
        trees_data.append({
            "cl": t.children_left.tolist(),
            "cr": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "th": [round(float(v), 4) for v in t.threshold],
            "v": leaf_classes,
        })

    payload = {
        "classes": classes,
        "trees": trees_data,
    }

    output_js_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_js_path, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated BHAVIK Client-Side Random Forest Model */\n")
        f.write("window.ASL_MODEL_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")

    size_mb = output_js_path.stat().st_size / (1024 * 1024)
    print(f"Exported client-side model to {output_js_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    m_path = Path(__file__).resolve().parent.parent / "models" / "asl_random_forest.joblib"
    out_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "js" / "asl_model_data.js"
    export_model_to_js(m_path, out_path, num_trees=50)
