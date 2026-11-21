# BHAVIK: Bridging Hands and Visual Intelligence through Knowledge
### Real-Time Sign Language Fingerspelling Recognition and Transcription System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00d2b4.svg)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-F7931E.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.17%25-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Pytest-32%20Passed-success.svg)]()

> *"Your Hands. Your Voice. Your Words."*

**BHAVIK** is a real-time, cross-platform sign language fingerspelling recognition and continuous transcription system engineered for second-year B.Tech Computer Science & Engineering students, researchers, and accessibility advocates.

The system recognizes the **24 static letters of the American Sign Language (ASL) alphabet** through live video feeds and continuously transcribes hand gestures into words and sentences. It is architected from the ground up as a responsive, mobile-first progressive web application that works seamlessly on both **desktop webcams** and **Android/iOS smartphone cameras**.

---

## 🚀 Key Features

- **Multi-Device Cross-Platform Support**: Delivers an optimal user experience across desktops, laptops, tablets, Android smartphones (Chrome), and iPhones (Safari).
- **Computer Vision Hand Tracking**: Uses Google MediaPipe Hands to detect and track 21 3D hand landmarks in real time on CPU.
- **Invariant Feature Engineering**: 63-dimensional feature space normalized for translation (wrist-centric) and scale (palm-span distance invariance).
- **Genuine Machine Learning**: Trained Scikit-Learn Random Forest Classifier (150 trees) achieving **96.17% test accuracy** and **0.85 ms CPU latency** on independent validation sets.
- **Temporal Prediction Stabilizer**: 10-frame sliding window majority vote with hysteresis and cooldown logic to eliminate frame flickers and prevent runaway repeated letters.
- **Continuous Transcription Engine**: Space insertion, backspace, clear, clipboard copy, and browser Text-to-Speech (TTS) vocalization.
- **Progressive Web App (PWA)**: Standalone installable experience with web app manifest and offline app shell caching.
- **Academic Rigor**: Clear scope boundaries: distinguishes ASL from ISL and documents dynamic motion letters (**J** and **Z**) as requiring trajectory modeling rather than simulated static prediction.

---

## 📂 Project Architecture

```
BHAVIK/
├── backend/
│   ├── main.py                     # FastAPI server entry point
│   ├── config.py                   # Central configuration & hyperparameters
│   ├── requirements.txt            # Dependency specification
│   ├── api/
│   │   ├── routes.py               # REST endpoints for health, metadata, data collection
│   │   └── websocket.py            # Real-time WebSocket recognition endpoint (/ws/recognize)
│   ├── src/
│   │   ├── hand_detector.py        # MediaPipe 21-landmark tracking module
│   │   ├── feature_extractor.py    # Invariant 63-feature transformation
│   │   ├── data_collector.py       # Dataset collection and CSV persistence
│   │   ├── model_trainer.py        # Scikit-learn Random Forest training pipeline
│   │   ├── predictor.py            # High-speed confidence-rated inference engine
│   │   ├── prediction_smoother.py  # Temporal stabilization & cooldown state machine
│   │   └── transcription.py        # Continuous sentence accumulation & editing
│   ├── scripts/
│   │   ├── collect_data.py         # Desktop interactive webcam data collector
│   │   ├── generate_asl_dataset.py # Canonical 3D biomechanical benchmark dataset generator
│   │   ├── train_model.py          # Single-command model training and verification
│   │   └── evaluate_model.py       # Comprehensive per-class evaluation & SVM comparison
│   ├── data/
│   │   ├── raw/                    # Raw session captures
│   │   └── processed/              # Curated CSV dataset (3,000 labelled samples)
│   └── models/
│       ├── asl_random_forest.joblib# Serialized production Random Forest model
│       └── model_metadata.json     # Architecture metadata & validation metrics
│
├── frontend/
│   ├── index.html                  # Accessible mobile-first web interface
│   ├── manifest.json               # Progressive Web App manifest
│   ├── service-worker.js           # PWA offline cache worker
│   ├── assets/                     # App icons (icon-192.png, icon-512.png)
│   ├── css/
│   │   └── style.css               # Responsive design system & dark navy theme
│   └── js/
│       ├── camera.js               # MediaDevices getUserMedia, dual-camera & mirroring
│       ├── websocket.js            # WebSocket client with auto-reconnection
│       ├── transcription.js        # Text buffer, clipboard, and Speech Synthesis (TTS)
│       └── app.js                  # Application glue code and HUD rendering
│
├── tests/
│   ├── test_features.py            # Feature extractor invariant mathematical tests
│   ├── test_smoother.py            # Temporal stabilization & cooldown state machine tests
│   ├── test_transcription.py       # Continuous transcription editing & spelling tests
│   ├── test_model.py               # Production model loading & probability tests
│   └── test_api.py                 # REST & WebSocket API integration tests
│
├── docs/
│   ├── architecture.md             # In-depth architectural specification & dataflow
│   ├── methodology.md              # Academic B.Tech project report & literature review
│   ├── presentation_notes.md       # PPT slides, 5/10-min scripts & viva questions
│   └── mobile_testing.md           # Multi-device testing guide & HTTPS tunnel setup
│
├── README.md                       # Comprehensive project documentation
└── .gitignore                      # Git ignore rules
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- Python 3.10 or 3.11 (Recommended for MediaPipe pre-built wheels on Windows/macOS/Linux)
- A modern web browser (Google Chrome, Microsoft Edge, Safari, Firefox)
- A built-in or external webcam

### 2. Environment Setup
```powershell
# Navigate to project directory
cd "BHAVIK"

# Create a dedicated virtual environment
uv venv .venv --python 3.10
# (Or using standard python: python -m venv .venv)

# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies
uv pip install -r backend/requirements.txt
```

### 3. Start the BHAVIK Application
```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open your desktop browser and navigate to:
👉 **`http://localhost:8000`**

Click **Start Camera**, grant permission, and begin signing!

---

## 📱 Mobile Testing & HTTPS Instructions

### Why HTTPS is Required on Smartphones
Modern mobile browsers strictly enforce the W3C Secure Contexts specification: `navigator.mediaDevices.getUserMedia` is disabled on unencrypted HTTP networks (e.g. `http://192.168.x.x:8000`).

To test on your Android or iOS smartphone across Wi-Fi:

### Recommended: Cloudflare Development Tunnel (Free, Instant, No Account Needed)
1. Download `cloudflared` from [Cloudflare](https://github.com/cloudflare/cloudflared/releases).
2. Start the BHAVIK backend locally on port 8000.
3. In a second terminal, run:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
4. Copy the secure HTTPS URL (e.g., `https://bhavik-demo.trycloudflare.com`) into Chrome on Android or Safari on iPhone. Both the camera and WebSocket (`wss://`) will connect seamlessly.

*For complete mobile testing checklists and alternative USB port-forwarding options, see [docs/mobile_testing.md](file:///c:/Users/jiten/imp%20files/BHAVIK/docs/mobile_testing.md).*

---

## 🧠 Machine Learning Model & Training

### Retrain the Model in a Single Command
To retrain the Random Forest model on the 3,000-sample dataset:
```powershell
.venv\Scripts\python.exe backend/scripts/train_model.py --estimators 150 --depth 20
```

### Detailed Evaluation & Baseline SVM Comparison
To evaluate per-class precision, recall, confusion matrix, and comparative performance against Support Vector Machines:
```powershell
.venv\Scripts\python.exe backend/scripts/evaluate_model.py
```

### Empirical Results Summary
- **Model**: `RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced')`
- **Classes**: 24 static ASL alphabet letters (`A`–`Y`, excluding motion letters `J` and `Z`)
- **Test Accuracy**: **96.17%**
- **Macro Precision**: **96.19%**
- **Macro F1-Score**: **96.17%**
- **Inference Latency**: **0.85 ms / frame** (Fast CPU inference)

---

## 🧪 Automated Test Suite

Run the full automated test suite covering all 32 unit and integration tests:
```powershell
.venv\Scripts\pytest.exe -v
```

Test coverage includes:
- `test_features.py`: Scale, translation invariance, wrist-at-origin, and 63-D shape validation.
- `test_smoother.py`: Sliding FIFO queue, 80% majority voting, cooldown timers, and held-gesture suppression.
- `test_transcription.py`: Continuous spelling sequence (*BHAVIK*), space handling, backspace, and undo.
- `test_model.py`: Model deserialization, probability distribution integrity, and metadata schema.
- `test_api.py`: FastAPI REST diagnostic endpoints, PWA assets, and WebSocket message handling.

---

## 📖 Academic Documentation

Comprehensive documentation prepared for academic presentations, college seminars, and project defense:
- [docs/architecture.md](file:///c:/Users/jiten/imp%20files/BHAVIK/docs/architecture.md): Component diagrams, dataflow, and WebSocket wire protocol.
- [docs/methodology.md](file:///c:/Users/jiten/imp%20files/BHAVIK/docs/methodology.md): 18-section research report, literature survey, and mathematical formulations.
- [docs/presentation_notes.md](file:///c:/Users/jiten/imp%20files/BHAVIK/docs/presentation_notes.md): 10-slide PPT outline, 5/10-minute presentation scripts, and viva voce Q&A handbook.
- [docs/mobile_testing.md](file:///c:/Users/jiten/imp%20files/BHAVIK/docs/mobile_testing.md): Cross-device QA matrix, touch target validation, and HTTPS deployment.

---

## ⚖️ Academic & Ethical Disclaimers

1. **Static Fingerspelling Limitation**: BHAVIK recognizes static fingerspelling letters. It is **not** a complete sign-language sentence translator.
2. **Exclusion of J and Z**: Letters **J** and **Z** inherently involve motion trajectories (rotational dip and zig-zag respectively). To preserve academic integrity, they are not falsely classified with a static spatial model.
3. **ASL vs. ISL**: American Sign Language is distinct from Indian Sign Language (ISL). ASL uses a one-handed fingerspelling alphabet; ISL frequently uses two hands.
4. **Assistive Tool**: BHAVIK is an educational and communication aid and does not replace certified human sign-language interpreters.

---

## 📚 External References

- Google MediaPipe Hands: [Hand Landmarker Documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- Sign Language ML Reference: [KhaledEisa/Sign-Language-ML](https://github.com/KhaledEisa/Sign-Language-ML)
- Real-Time ASL Detection: [zyle-geralde/ASLDetectionMediaPipe](https://github.com/zyle-geralde/ASLDetectionMediaPipe)
- FastAPI Framework: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- W3C Media Capture and Streams: [MediaDevices getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- Mozilla MDN: [Progressive Web Apps (PWA)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
