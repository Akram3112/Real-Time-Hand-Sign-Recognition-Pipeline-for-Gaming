import cv2
import mediapipe as mp
import numpy as np
import joblib

# ==============================
# LOAD CLASSIFIER
# ==============================

# In case you wanna test the DT on the unprocessed dataset, make sure to load the corresponding model:
# bundle      = joblib.load("model/model_dt_notAugmented.joblib")
bundle      = joblib.load("model/model_dt.joblib")
clf         = bundle["model"]
CLASS_NAMES = bundle["class_names"]

# ==============================
# MEDIAPIPE
# ==============================

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
HandConnections       = mp.tasks.vision.HandLandmarksConnections

# ==============================
# FEATURE EXTRACTION
# must match feature_extractor.py exactly:
# no recentering, world landmarks as-is
# left 63 + right 63 = 126 features
# ==============================

def hand_shape(world_lms):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_lms])  # (21, 3)
    return pts.flatten()  # 63 values, no recentering

def extract_features(result):
    left_feat  = np.zeros(63)
    right_feat = np.zeros(63)

    if result.hand_landmarks:
        for i in range(min(len(result.hand_landmarks), 2)):
            side      = result.handedness[i][0].category_name
            world_lms = result.hand_world_landmarks[i]
            features  = hand_shape(world_lms)
            if side == "Left":
                left_feat  = features
            else:
                right_feat = features

    return np.concatenate([left_feat, right_feat]).reshape(1, -1)

# ==============================
# DRAW LANDMARKS
# ==============================

def draw_landmarks(rgb_image, detection_result):
    out = np.copy(rgb_image)
    if not detection_result.hand_landmarks:
        return out
    h, w, _ = out.shape
    for hand_landmarks in detection_result.hand_landmarks:
        for lm in hand_landmarks:
            cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
        for conn in HandConnections.HAND_CONNECTIONS:
            s = hand_landmarks[conn.start]
            e = hand_landmarks[conn.end]
            cv2.line(out,
                     (int(s.x * w), int(s.y * h)),
                     (int(e.x * w), int(e.y * h)),
                     (255, 0, 0), 2)
    return out

# ==============================
# PREDICTION SMOOTHING
# majority vote over last N frames
# to stabilize the displayed label
# ==============================

HISTORY_SIZE = 15
history      = []

def smooth_predict(features):
    pred = clf.predict(features)[0]
    history.append(pred)
    if len(history) > HISTORY_SIZE:
        history.pop(0)
    return max(set(history), key=history.count)

# ==============================
# MEDIAPIPE OPTIONS
# ==============================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = HandLandmarker.create_from_options(options)

# ==============================
# CAMERA LOOP
# ==============================

cap      = cv2.VideoCapture(0)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # mirror view — matches recording conditions
    frame    = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((frame_id / 30) * 1000)
    result       = detector.detect_for_video(mp_image, timestamp_ms)

    # ── draw skeleton ─────────────────────────────────────────────
    annotated = draw_landmarks(rgb, result)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # ── extract features ──────────────────────────────────────────
    features = extract_features(result)

    left_detected  = (features[0, :63]  != 0).any()
    right_detected = (features[0, 63:]  != 0).any()
    both_detected  = left_detected and right_detected

    # ── classify ──────────────────────────────────────────────────
    if both_detected:
        class_id   = smooth_predict(features)
        label_text = CLASS_NAMES[class_id].upper()
        color      = (0, 220, 0)
    elif left_detected or right_detected:
        # one hand visible — still try to classify
        # (some gestures have one hand missing by nature)
        class_id   = smooth_predict(features)
        label_text = CLASS_NAMES[class_id].upper() + " (?)"
        color      = (0, 180, 255)
    else:
        label_text = "NO HANDS DETECTED"
        color      = (0, 0, 255)

    # ── overlay label ─────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    thickness  = 3
    margin     = 12

    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(annotated,
                  (margin, margin),
                  (margin + tw + 16, margin + th + baseline + 16),
                  (0, 0, 0), -1)
    cv2.putText(annotated, label_text,
                (margin + 8, margin + th + 8),
                font, font_scale, color, thickness, cv2.LINE_AA)

    # ── detection status ──────────────────────────────────────────
    status = f"L: {'OK' if left_detected else '--'}  R: {'OK' if right_detected else '--'}"
    cv2.putText(annotated, status,
                (margin + 8, margin + th + baseline + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Naruto Hand Sign — DT", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()