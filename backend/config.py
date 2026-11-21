import os
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BACKEND_DIR / "models"
CERTS_DIR = BACKEND_DIR / "certs"

# Ensure essential directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CERTS_DIR.mkdir(parents=True, exist_ok=True)

# Server Configuration
HOST = os.getenv("BHAVIK_HOST", "0.0.0.0")
PORT = int(os.getenv("BHAVIK_PORT", "8000"))
HTTPS_PORT = int(os.getenv("BHAVIK_HTTPS_PORT", "8443"))
DEBUG = os.getenv("BHAVIK_DEBUG", "false").lower() == "true"

# Computer Vision & MediaPipe Configuration
MIN_DETECTION_CONFIDENCE = float(os.getenv("BHAVIK_DETECTION_CONFIDENCE", "0.5"))
MIN_TRACKING_CONFIDENCE = float(os.getenv("BHAVIK_TRACKING_CONFIDENCE", "0.5"))
MAX_NUM_HANDS = 1

# Supported ASL Fingerspelling Classes (Full 26-Letter Alphabet A through Z)
SUPPORTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]
EXCLUDED_CLASSES = {}

# Machine Learning Paths
MODEL_PATH = MODELS_DIR / "asl_random_forest.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
_DATASET_GZ = PROCESSED_DATA_DIR / "asl_landmarks_dataset.csv.gz"
DATASET_CSV_PATH = _DATASET_GZ if _DATASET_GZ.exists() else (PROCESSED_DATA_DIR / "asl_landmarks_dataset.csv")

# Prediction & Stabilization Defaults (Tuned for smooth, responsive real-time interaction)
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_SMOOTHING_BUFFER_SIZE = 7
DEFAULT_STABILITY_RATIO = 0.70
DEFAULT_COOLDOWN_SECONDS = 0.9
