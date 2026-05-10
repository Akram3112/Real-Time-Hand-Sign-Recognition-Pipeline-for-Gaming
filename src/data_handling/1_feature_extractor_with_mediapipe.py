import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import re
import pandas as pd

# ==============================
# CONFIG
# ==============================

DATASET_PATH = "../../1_dataset_VIDEO"
MODEL_PATH   = "../../model/hand_landmarker.task"
OUTPUT_PATH  = "../../2_features"

os.makedirs(OUTPUT_PATH, exist_ok=True)

SKIP_SECONDS = 2

# ==============================
# MEDIAPIPE
# ==============================

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ==============================
# HEADER
#
# 126 features total (+ 1 person_id = 127 cols):
#   left_*  : 63  (21 world landmarks x 3 coords, wrist at origin)
#   right_* : 63  (same)
#
# No horizontal flip applied — MediaPipe's handedness model is trained
# for selfie/mirror view which matches typical recording conditions.
# Flipping would corrupt the Left/Right labels.
# ==============================

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip",  "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp","middle_pip","middle_dip","middle_tip",
    "ring_mcp",  "ring_pip",  "ring_dip",  "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

header = ["person_id"]
for hand in ["left", "right"]:
    for lm in LANDMARK_NAMES:
        for c in ["x", "y", "z"]:
            header.append(f"{hand}_{lm}_{c}")

# ==============================
# FEATURE HELPER
# ==============================

def hand_shape(world_lms):
    """
    63 features for one hand from world landmarks.

    World landmarks are in real-world meters. We re-center on the
    wrist (index 0) so it sits exactly at (0,0,0). This makes the
    features invariant to where the hand is in the frame and how
    far it is from the camera.

    Returns a flat list of 63 floats.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_lms])  # (21,3)
    return pts.flatten().tolist()

# ==============================
# PROCESS DATASET
# ==============================

for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print("\nProcessing label:", label)

    output_file = os.path.join(OUTPUT_PATH, f"{label}.csv")

    # ── skip already processed persons ─────────────────────────────
    existing_persons = set()
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        if "person_id" in df.columns:
            existing_persons = set(df["person_id"].unique())

    # ── find best take per person ───────────────────────────────────
    best_take = {}
    for video_name in os.listdir(label_path):
        match = re.search(r"_p(\d+)_t(\d+)", video_name)
        if not match:
            continue
        person_id = int(match.group(1))
        take      = int(match.group(2))
        if person_id not in best_take or take > best_take[person_id][1]:
            best_take[person_id] = (video_name, take)

    # ── process videos ──────────────────────────────────────────────
    rows = []

    for person_id, (video_name, take) in best_take.items():

        if person_id in existing_persons:
            print(f"   Person {person_id} already exists -> skipping")
            continue

        video_path = os.path.join(label_path, video_name)
        print(f"   Processing person {person_id} take {take}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )

        detector = HandLandmarker.create_from_options(options)
        cap      = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        skip_frames = fps * SKIP_SECONDS
        frame_id    = 0

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id < skip_frames:
                continue

            # No flip — MediaPipe handedness is correct for selfie/mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_id / fps) * 1000)
            result       = detector.detect_for_video(mp_image, timestamp_ms)

            left_feat  = [0.0] * 63   # zeros = hand not detected
            right_feat = [0.0] * 63

            if result.hand_landmarks:
                for i in range(min(len(result.hand_landmarks), 2)):
                    side      = result.handedness[i][0].category_name
                    world_lms = result.hand_world_landmarks[i]
                    features  = hand_shape(world_lms)
                    if side == "Left":
                        left_feat  = features
                    else:
                        right_feat = features

            row = [person_id] + left_feat + right_feat
            rows.append(row)

        cap.release()

    # ── save / append ───────────────────────────────────────────────
    write_header = not os.path.exists(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)

    print("Saved / updated:", output_file)