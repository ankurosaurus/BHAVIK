"""
BHAVIK WebSocket Recognition Endpoint
Handles low-latency real-time video frame streaming, landmark tracking,
ML inference, temporal smoothing, and continuous transcription updates.
"""

import base64
import json
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.src.feature_extractor import FeatureExtractor
from backend.src.hand_detector import HandDetector
from backend.src.prediction_smoother import PredictionSmoother
from backend.src.predictor import ASLPredictor
from backend.src.transcription import TranscriptionEngine

router = APIRouter(tags=["WebSocket"])
shared_predictor = ASLPredictor()


@router.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()

    # Per-connection session components
    detector = HandDetector(static_image_mode=False)
    extractor = FeatureExtractor()
    smoother = PredictionSmoother()
    transcription = TranscriptionEngine()

    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                msg = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            # Handle control actions from frontend
            action = msg.get("action")
            if action:
                if action == "space":
                    transcription.add_space()
                elif action == "backspace":
                    transcription.backspace()
                elif action == "clear":
                    transcription.clear()
                elif action == "undo":
                    transcription.undo()
                elif action == "configure":
                    settings = msg.get("settings", {})
                    smoother.configure(
                        buffer_size=settings.get("buffer_size"),
                        stability_ratio=settings.get("stability_ratio"),
                        confidence_threshold=settings.get("confidence_threshold"),
                        cooldown_seconds=settings.get("cooldown_seconds"),
                    )

                # Send updated transcription state back immediately
                await websocket.send_json({
                    "action_ack": action,
                    "transcription": transcription.get_state(),
                })
                continue

            # Process frame
            image_data = msg.get("image")
            if not image_data:
                continue

            # Strip Data URI prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({
                    "hand_detected": False,
                    "status": "DECODE_ERROR",
                    "transcription": transcription.get_state(),
                })
                continue

            if frame is None or frame.size == 0:
                await websocket.send_json({
                    "hand_detected": False,
                    "status": "EMPTY_FRAME",
                    "transcription": transcription.get_state(),
                })
                continue

            # FPS calculation
            fps_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                current_fps = fps_count / (now - fps_start_time)
                fps_count = 0
                fps_start_time = now

            # Computer vision & inference
            hand_detected = detector.process_frame(frame)
            landmarks = None
            pred_letter: Optional[str] = None
            confidence: float = 0.0

            if hand_detected:
                landmarks = detector.get_landmarks(0)
                if landmarks:
                    features = extractor.extract_from_dict_list(landmarks)
                    if features is not None:
                        pred_letter, confidence, _ = shared_predictor.predict(features)

            smoother_res = smoother.process(
                predicted_letter=pred_letter,
                confidence=confidence,
                hand_detected=hand_detected,
                current_time=now,
            )

            # If smoother confirmed a new accepted letter, append to transcription
            if smoother_res.accepted_letter:
                transcription.add_letter(smoother_res.accepted_letter)

            response = {
                "hand_detected": hand_detected,
                "landmarks": landmarks,
                "predicted_letter": smoother_res.predicted_letter,
                "confidence": round(smoother_res.confidence, 3),
                "is_stable": smoother_res.is_stable,
                "stabilization_progress": round(smoother_res.stability_progress, 2),
                "accepted_letter": smoother_res.accepted_letter,
                "status": smoother_res.status,
                "transcription": transcription.get_state(),
                "server_fps": round(current_fps, 1),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session error: {e}")
    finally:
        detector.close()
