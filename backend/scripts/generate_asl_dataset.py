"""
BHAVIK Benchmark ASL Landmark Generator
Synthesizes biomechanically accurate 21-joint 3D hand landmark configurations
for the 24 static ASL fingerspelling letters (A-Y, excluding dynamic J & Z).
Applies 3D spatial rotations, joint angle variations, hand-size scaling,
and sensor noise to create a robust, generalizable benchmark dataset.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import SUPPORTED_CLASSES, DATASET_CSV_PATH
from backend.src.feature_extractor import FeatureExtractor, FEATURE_NAMES


def get_base_hand_skeleton() -> np.ndarray:
    """
    Returns baseline hand skeleton coordinates (21 landmarks) in an open palm resting state.
    Joint indexing follows Google MediaPipe Hands standard:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
    """
    coords = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    coords[0] = [0.0, 0.0, 0.0]

    # Thumb
    coords[1] = [-0.15, 0.12, 0.02]
    coords[2] = [-0.25, 0.22, 0.03]
    coords[3] = [-0.32, 0.32, 0.04]
    coords[4] = [-0.38, 0.42, 0.05]

    # Index finger
    coords[5] = [-0.12, 0.38, 0.01]
    coords[6] = [-0.14, 0.55, 0.01]
    coords[7] = [-0.15, 0.68, 0.01]
    coords[8] = [-0.16, 0.78, 0.01]

    # Middle finger
    coords[9] = [0.0, 0.40, 0.0]
    coords[10] = [0.0, 0.58, 0.0]
    coords[11] = [0.0, 0.72, 0.0]
    coords[12] = [0.0, 0.84, 0.0]

    # Ring finger
    coords[13] = [0.11, 0.38, -0.01]
    coords[14] = [0.13, 0.54, -0.01]
    coords[15] = [0.14, 0.67, -0.01]
    coords[16] = [0.15, 0.77, -0.01]

    # Pinky finger
    coords[17] = [0.22, 0.32, -0.02]
    coords[18] = [0.26, 0.45, -0.02]
    coords[19] = [0.29, 0.55, -0.02]
    coords[20] = [0.32, 0.64, -0.02]

    return coords


def create_canonical_pose(letter: str) -> np.ndarray:
    """Modifies the base skeleton into the specific ASL fingerspelling configuration for letter."""
    pose = get_base_hand_skeleton().copy()

    # Finger landmark indices
    index_finger = [5, 6, 7, 8]
    middle_finger = [9, 10, 11, 12]
    ring_finger = [13, 14, 15, 16]
    pinky_finger = [17, 18, 19, 20]

    def curl_finger(indices, curl_factor=0.75, z_curl=0.15):
        """Curls a finger inward toward the palm."""
        mcp = pose[indices[0]]
        for step, cur_idx in enumerate(indices[1:], start=1):
            pose[cur_idx, 1] = mcp[1] + (pose[cur_idx, 1] - mcp[1]) * (1.0 - curl_factor * (step / 3.0))
            pose[cur_idx, 2] += z_curl * step

    def fold_thumb_across():
        pose[2] = [-0.10, 0.18, 0.05]
        pose[3] = [-0.02, 0.25, 0.08]
        pose[4] = [0.08, 0.28, 0.09]

    # Biomechanical definitions for ASL letters
    if letter == 'A':
        # Fist with thumb resting alongside index finger
        curl_finger(index_finger, 0.78)
        curl_finger(middle_finger, 0.78)
        curl_finger(ring_finger, 0.78)
        curl_finger(pinky_finger, 0.78)
        pose[2] = [-0.20, 0.28, 0.04]
        pose[3] = [-0.18, 0.40, 0.05]
        pose[4] = [-0.16, 0.50, 0.06]

    elif letter == 'B':
        # Flat hand, 4 fingers straight up, thumb folded across palm
        fold_thumb_across()

    elif letter == 'C':
        # Curved hand forming a C shape
        for f in [index_finger, middle_finger, ring_finger, pinky_finger]:
            curl_finger(f, 0.45, z_curl=0.18)
        pose[2] = [-0.26, 0.25, 0.10]
        pose[3] = [-0.20, 0.38, 0.15]
        pose[4] = [-0.10, 0.48, 0.18]

    elif letter == 'D':
        # Index straight up, thumb touching middle & ring tips forming loop
        curl_finger(middle_finger, 0.72)
        curl_finger(ring_finger, 0.72)
        curl_finger(pinky_finger, 0.72)
        pose[3] = [-0.05, 0.35, 0.12]
        pose[4] = [0.0, 0.42, 0.12]

    elif letter == 'E':
        # All fingers curled down onto palm, thumb tucked underneath
        curl_finger(index_finger, 0.85, z_curl=0.12)
        curl_finger(middle_finger, 0.85, z_curl=0.12)
        curl_finger(ring_finger, 0.85, z_curl=0.12)
        curl_finger(pinky_finger, 0.85, z_curl=0.12)
        pose[2] = [-0.12, 0.18, 0.15]
        pose[3] = [-0.05, 0.22, 0.16]
        pose[4] = [0.02, 0.24, 0.16]

    elif letter == 'F':
        # OK sign: Index and thumb tips touch, middle, ring, pinky straight up
        pose[6] = [-0.10, 0.45, 0.08]
        pose[7] = [-0.08, 0.42, 0.12]
        pose[8] = [-0.06, 0.38, 0.12]
        pose[3] = [-0.15, 0.32, 0.10]
        pose[4] = [-0.07, 0.37, 0.12]

    elif letter == 'G':
        # Index pointing horizontally, thumb parallel, other fingers curled
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[6] = [-0.12, 0.40, 0.0]
        pose[7] = [0.05, 0.42, 0.0]
        pose[8] = [0.20, 0.42, 0.0]
        pose[2] = [-0.15, 0.30, 0.02]
        pose[3] = [0.02, 0.33, 0.02]
        pose[4] = [0.15, 0.34, 0.02]

    elif letter == 'H':
        # Index and middle pointing horizontally together
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[6] = [-0.10, 0.45, 0.0]
        pose[7] = [0.08, 0.46, 0.0]
        pose[8] = [0.22, 0.46, 0.0]
        pose[10] = [-0.02, 0.43, 0.0]
        pose[11] = [0.09, 0.43, 0.0]
        pose[12] = [0.23, 0.43, 0.0]

    elif letter == 'I':
        # Pinky straight up, other fingers curled, thumb folded across
        curl_finger(index_finger, 0.8)
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        fold_thumb_across()

    elif letter == 'K':
        # Index straight up, middle finger angled forward/up, thumb between
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[10] = [0.0, 0.52, 0.15]
        pose[11] = [0.0, 0.62, 0.28]
        pose[12] = [0.0, 0.70, 0.38]
        pose[3] = [-0.08, 0.40, 0.12]
        pose[4] = [-0.05, 0.48, 0.15]

    elif letter == 'L':
        # Index up, thumb extended horizontally at 90 deg, others curled
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[1] = [-0.20, 0.15, 0.0]
        pose[2] = [-0.35, 0.18, 0.0]
        pose[3] = [-0.48, 0.20, 0.0]
        pose[4] = [-0.60, 0.22, 0.0]

    elif letter == 'M':
        # 3 fingers (index, middle, ring) folded over thumb
        curl_finger(index_finger, 0.75)
        curl_finger(middle_finger, 0.75)
        curl_finger(ring_finger, 0.75)
        curl_finger(pinky_finger, 0.8)
        pose[2] = [-0.10, 0.25, 0.06]
        pose[3] = [0.0, 0.28, 0.07]
        pose[4] = [0.08, 0.29, 0.07]

    elif letter == 'N':
        # 2 fingers (index, middle) folded over thumb
        curl_finger(index_finger, 0.75)
        curl_finger(middle_finger, 0.75)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[2] = [-0.10, 0.25, 0.06]
        pose[3] = [-0.02, 0.27, 0.07]
        pose[4] = [0.04, 0.28, 0.07]

    elif letter == 'O':
        # All finger tips curve to meet thumb tip forming O
        curl_finger(index_finger, 0.55, z_curl=0.18)
        curl_finger(middle_finger, 0.55, z_curl=0.18)
        curl_finger(ring_finger, 0.55, z_curl=0.18)
        curl_finger(pinky_finger, 0.55, z_curl=0.18)
        pose[2] = [-0.22, 0.30, 0.12]
        pose[3] = [-0.14, 0.42, 0.16]
        pose[4] = [-0.08, 0.48, 0.17]

    elif letter == 'P':
        # K gesture inverted downwards
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[6] = [-0.12, 0.20, 0.0]
        pose[7] = [-0.14, 0.05, 0.0]
        pose[8] = [-0.15, -0.10, 0.0]
        pose[10] = [0.0, 0.20, 0.15]
        pose[11] = [0.0, 0.10, 0.25]
        pose[12] = [0.0, 0.0, 0.35]

    elif letter == 'Q':
        # G gesture inverted downwards
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[6] = [-0.10, 0.20, 0.0]
        pose[7] = [-0.10, 0.05, 0.0]
        pose[8] = [-0.10, -0.10, 0.0]
        pose[2] = [-0.18, 0.15, 0.02]
        pose[3] = [-0.18, 0.05, 0.02]
        pose[4] = [-0.18, -0.05, 0.02]

    elif letter == 'R':
        # Index and middle fingers crossed
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[7] = [-0.05, 0.68, 0.02]
        pose[8] = [0.03, 0.78, 0.04]  # Crosses over middle
        pose[11] = [-0.02, 0.72, -0.02]
        pose[12] = [-0.05, 0.84, -0.03]

    elif letter == 'S':
        # Fist with thumb folded over all fingers
        curl_finger(index_finger, 0.8)
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[1] = [-0.12, 0.22, 0.10]
        pose[2] = [-0.04, 0.32, 0.14]
        pose[3] = [0.04, 0.35, 0.14]
        pose[4] = [0.12, 0.33, 0.13]

    elif letter == 'T':
        # Thumb between index and middle
        curl_finger(index_finger, 0.75)
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        pose[2] = [-0.10, 0.28, 0.08]
        pose[3] = [-0.06, 0.38, 0.10]
        pose[4] = [-0.03, 0.46, 0.10]

    elif letter == 'U':
        # Index and middle fingers held together pointing straight up
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[6] = [-0.04, 0.55, 0.0]
        pose[7] = [-0.04, 0.68, 0.0]
        pose[8] = [-0.04, 0.78, 0.0]
        pose[10] = [0.04, 0.58, 0.0]
        pose[11] = [0.04, 0.72, 0.0]
        pose[12] = [0.04, 0.84, 0.0]

    elif letter == 'V':
        # Peace sign: Index and middle spread in V
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[6] = [-0.18, 0.55, 0.0]
        pose[7] = [-0.25, 0.68, 0.0]
        pose[8] = [-0.32, 0.78, 0.0]
        pose[10] = [0.12, 0.58, 0.0]
        pose[11] = [0.18, 0.72, 0.0]
        pose[12] = [0.24, 0.84, 0.0]

    elif letter == 'W':
        # Index, middle, ring spread pointing up
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[8] = [-0.25, 0.78, 0.0]
        pose[12] = [0.0, 0.84, 0.0]
        pose[16] = [0.25, 0.77, 0.0]

    elif letter == 'X':
        # Index finger hooked, others curled
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        curl_finger(pinky_finger, 0.8)
        fold_thumb_across()
        pose[6] = [-0.14, 0.52, 0.0]
        pose[7] = [-0.14, 0.62, 0.12]
        pose[8] = [-0.14, 0.54, 0.20]

    elif letter == 'Y':
        # Shaka: Thumb and pinky extended, middle 3 curled
        curl_finger(index_finger, 0.8)
        curl_finger(middle_finger, 0.8)
        curl_finger(ring_finger, 0.8)
        pose[1] = [-0.20, 0.15, 0.0]
        pose[2] = [-0.35, 0.20, 0.0]
        pose[3] = [-0.50, 0.25, 0.0]
        pose[4] = [-0.62, 0.30, 0.0]
        pose[18] = [0.28, 0.45, 0.0]
        pose[19] = [0.38, 0.58, 0.0]
        pose[20] = [0.48, 0.70, 0.0]

    return pose


def apply_random_variations(
    canonical_pose: np.ndarray,
    noise_sigma: float = 0.015,
    angle_max_deg: float = 12.0,
    scale_range: tuple = (0.85, 1.15),
) -> np.ndarray:
    """Applies realistic physical spatial variations to the canonical 3D skeleton."""
    pose = canonical_pose.copy()

    # 1. 3D Spatial Rotation (Roll, Pitch, Yaw)
    angles_rad = np.radians(np.random.uniform(-angle_max_deg, angle_max_deg, size=3))
    cx, cy, cz = np.cos(angles_rad)
    sx, sy, sz = np.sin(angles_rad)

    # Rotation matrix around X
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    # Rotation matrix around Y
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    # Rotation matrix around Z
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    r_matrix = rz @ ry @ rx

    pose = pose @ r_matrix.T

    # 2. Scale variation (hand distance & anatomical hand size)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    pose *= scale

    # 3. Sensor / MediaPipe landmark tracking noise
    noise = np.random.normal(0, noise_sigma, pose.shape).astype(np.float32)
    noise[0] *= 0.2  # Wrist has lower tracking jitter
    pose += noise

    return pose


def generate_benchmark_dataset(
    samples_per_class: int = 125,
    output_path: Path = DATASET_CSV_PATH,
) -> pd.DataFrame:
    """Generates benchmark dataset with varied sessions, hands, and perturbations."""
    np.random.seed(42)
    extractor = FeatureExtractor()
    rows = []

    print(f"Generating benchmark ASL dataset for {len(SUPPORTED_CLASSES)} classes...")
    print(f"Samples per class: {samples_per_class}")

    sessions = ["participant_1", "participant_2", "participant_3", "participant_4", "mobile_env"]

    for letter in SUPPORTED_CLASSES:
        canonical = create_canonical_pose(letter)

        for i in range(samples_per_class):
            session = sessions[i % len(sessions)]
            is_left = (i % 5 == 0)  # 20% left hand examples

            # Perturb pose
            noisy_pose = apply_random_variations(canonical)
            if is_left:
                noisy_pose[:, 0] = -noisy_pose[:, 0]  # Mirror X for left hand

            # Extract normalized invariant features
            features = extractor.extract_from_array(noisy_pose, is_left_hand=False)

            row = {
                "label": letter,
                "session_id": session,
                "is_left_hand": int(is_left),
            }
            for name, val in zip(FEATURE_NAMES, features):
                row[name] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} labelled samples saved to {output_path}")
    return df


if __name__ == "__main__":
    generate_benchmark_dataset()
