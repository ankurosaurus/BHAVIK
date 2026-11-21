"""
BHAVIK Desktop Data Collection Script
Allows interactive collection of ASL fingerspelling landmark samples using a local webcam.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
from backend.config import SUPPORTED_CLASSES
from backend.src.data_collector import DataCollector
from backend.src.hand_detector import HandDetector


def run_collector(target_letter: str, target_count: int, camera_index: int = 0):
    target_letter = target_letter.upper()
    if target_letter not in SUPPORTED_CLASSES:
        print(f"Error: Letter '{target_letter}' is not in supported classes: {SUPPORTED_CLASSES}")
        return

    collector = DataCollector()
    detector = HandDetector(static_image_mode=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    print("=" * 60)
    print(f"BHAVIK Data Collection: Letter '{target_letter}'")
    print(f"Target Samples: {target_count}")
    print("Controls:")
    print("  'c' - Capture single sample")
    print("  'a' - Toggle auto-capture mode (captures every 0.3s)")
    print("  'p' - Pause / Resume")
    print("  'q' - Quit collection")
    print("=" * 60)

    current_counts = collector.get_sample_counts()
    collected_this_session = 0
    auto_mode = False
    last_auto_capture = 0.0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Empty camera frame received.")
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            hand_detected = detector.process_frame(frame)
            annotated_frame = detector.draw_landmarks(frame) if hand_detected else frame

            # Auto-capture logic
            now = time.time()
            if auto_mode and hand_detected and (now - last_auto_capture >= 0.3):
                landmarks = detector.get_landmarks(0)
                if landmarks:
                    collector.add_sample(target_letter, landmarks, session_id="webcam_auto")
                    collected_this_session += 1
                    last_auto_capture = now
                    print(f"[{target_letter}] Collected {collected_this_session}/{target_count}")

            # On-screen HUD
            status_text = f"Letter: {target_letter} | Hand: {'DETECTED' if hand_detected else 'NOT DETECTED'}"
            count_text = f"Total in DB: {current_counts.get(target_letter, 0) + collected_this_session} | Session: {collected_this_session}/{target_count}"
            mode_text = f"Mode: {'AUTO-CAPTURE (ON)' if auto_mode else 'MANUAL (Press C)'}"

            cv2.putText(annotated_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
            cv2.putText(annotated_frame, count_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, mode_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if auto_mode else (200, 200, 200), 2)

            cv2.imshow("BHAVIK Landmark Collector", annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and hand_detected:
                landmarks = detector.get_landmarks(0)
                if landmarks:
                    collector.add_sample(target_letter, landmarks, session_id="webcam_manual")
                    collected_this_session += 1
                    print(f"[{target_letter}] Captured sample #{collected_this_session}")
            elif key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto-capture mode: {'ON' if auto_mode else 'OFF'}")

            if collected_this_session >= target_count:
                print(f"Target count of {target_count} reached for '{target_letter}'!")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"Session ended. Total samples saved for '{target_letter}': {collector.get_sample_counts().get(target_letter, 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ASL Hand Landmark Data")
    parser.add_argument("--letter", type=str, default="A", help="ASL letter to collect (A-Y)")
    parser.add_argument("--samples", type=int, default=100, help="Target number of samples")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()
    run_collector(args.letter, args.samples, args.camera)
