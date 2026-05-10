import math
import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# Add src to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visual_effect.naruto_effects import FireEffect, LightningEffect, ShadowCloneEffect, WaterEffect


# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
REFERENCE_PATH = os.path.join(PROJECT_ROOT, "mudra_reference")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "hand_landmarker.task")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "model", "mudra_mlp_frame_level.pt")
DT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_dt.joblib")
HEADER_ICON_PATH = os.path.join(REFERENCE_PATH, "naruto_no_bg.png")

WINDOW_W = 1335
WINDOW_H = 720
CAMERA_INDEX = 0
NUM_HANDS = 2
CONF_THRESHOLD = 0.45
FINAL_CONF_THRESHOLD = 0.50
PREPARE_SECONDS = 1
RECORD_SECONDS = 2
RESULT_SECONDS = 2
MIN_VALID_FRAMES = 5
CENTER_PANEL = (466, 76, 554, 516)
CENTER_CONTENT = (471, 105, 544, 484)

BG = (7, 10, 13)
PANEL = (12, 16, 20)
PANEL_2 = (17, 22, 28)
BORDER = (56, 67, 76)
ORANGE = (0, 95, 255)
FIRE = (0, 70, 255)
YELLOW = (0, 225, 255)
GREEN = (70, 220, 65)
WHITE = (236, 236, 236)
MUTED = (150, 154, 162)
RED = (35, 45, 235)
CYAN = (255, 180, 80)
PURPLE = (220, 75, 210)

MUDRAS = [
    "boar", "ram", "tiger", "rat", "dog", "bird",
    "ox", "hare", "dragon", "snake", "horse", "monkey"
]

JUTSUS = [
    {
        "name": "FIREBALL JUTSU",
        "sequence": ["snake", "ram", "monkey", "boar", "horse", "tiger"],
        "color": ORANGE,
        "symbol": "F",
    },
    {
        "name": "LIGHTNING BLADE",
        "sequence": ["ox", "hare", "monkey"],
        "color": WHITE,
        "symbol": "L",
    },
    {
        "name": "WATER DRAGON JUTSU",
        "sequence": ["tiger", "snake", "rat", "monkey", "bird", "dog"],
        "color": CYAN,
        "symbol": "W",
    },
    {
        "name": "SHADOW CLONE JUTSU",
        "sequence": ["ram", "snake", "tiger"],
        "color": PURPLE,
        "symbol": "S",
    },

]


# ==============================
# JUTSU EFFECT HELPERS
# ==============================

def jutsu_index_from_key(key):
    if ord("1") <= key <= ord(str(min(9, len(JUTSUS)))):
        idx = key - ord("1")
        if idx < len(JUTSUS):
            return idx

    key_char = chr(key).lower() if 0 <= key <= 255 else ""
    for idx, jutsu in enumerate(JUTSUS):
        if key_char == jutsu["name"][0].lower():
            return idx

    return None


def make_jutsu_effects(width, height):
    return {
        "F": FireEffect(),
        "L": LightningEffect(),
        "W": WaterEffect(width, height),
        "S": ShadowCloneEffect(width, height, num_clones=2),
    }


def reset_jutsu_effects(effects):
    for effect in effects.values():
        effect.reset()


def feed_jutsu_effects(effects, frame):
    for effect in effects.values():
        feed = getattr(effect, "feed", None)
        if feed is not None:
            feed(frame)


def trigger_jutsu_effect(effect, width, height):
    cx = width // 2
    cy = int(height * 0.58)

    if isinstance(effect, ShadowCloneEffect):
        effect.trigger()
    else:
        effect.trigger(cx, cy)


def draw_jutsu_effect(effect, frame):
    h, w = frame.shape[:2]
    cx = w // 2
    cy = int(h * 0.58)

    if isinstance(effect, ShadowCloneEffect):
        return effect.draw(frame)
    return effect.draw(frame, cx, cy)


# ==============================
# MEDIAPIPE AND MODEL SETUP
# ==============================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MudraMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# FEATURE EXTRACTION
# ==============================

def compute_finger_angles(coords_flat: np.ndarray) -> np.ndarray:
    if np.allclose(coords_flat, 0.0):
        return np.zeros(10, dtype=np.float32)

    coords_3d = np.array(coords_flat, dtype=np.float32).reshape(21, 3)
    finger_joints = [
        [1, 2, 3], [2, 3, 4],
        [5, 6, 7], [6, 7, 8],
        [9, 10, 11], [10, 11, 12],
        [13, 14, 15], [14, 15, 16],
        [17, 18, 19], [18, 19, 20],
    ]

    angles = []
    for a, b, c in finger_joints:
        v1 = coords_3d[a] - coords_3d[b]
        v2 = coords_3d[c] - coords_3d[b]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        cos_a = np.dot(v1, v2) / denom
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    return np.array(angles, dtype=np.float32)


def normalize_one_hand(values: np.ndarray) -> np.ndarray:
    coords_3d = np.array(values, dtype=np.float32).reshape(21, 3).copy()
    wrist = coords_3d[0].copy()
    coords_3d -= wrist
    scale = np.linalg.norm(coords_3d[12])
    if scale > 1e-8:
        coords_3d /= scale
    return coords_3d.reshape(-1)


def extract_feature_vector(result) -> np.ndarray:
    features = np.zeros(126, dtype=np.float32)

    if result.hand_landmarks and result.hand_world_landmarks:
        hands = list(zip(result.hand_landmarks, result.hand_world_landmarks))
        hands.sort(key=lambda h: h[0][0].x)

        for i, (_, world_hand) in enumerate(hands[:2]):
            offset = i * 63
            for j, lm in enumerate(world_hand):
                features[offset + j * 3 + 0] = lm.x
                features[offset + j * 3 + 1] = lm.y
                features[offset + j * 3 + 2] = lm.z

    return features


# ==============================
# MODEL LOADING AND PREDICTION
# ==============================

def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mlp_predictor(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)

    input_dim = int(checkpoint["input_dim"])
    class_names = list(checkpoint["class_names"])
    scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
    scaler_scale[scaler_scale == 0] = 1.0

    model = MudraMLP(input_dim=input_dim, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, scaler_mean, scaler_scale


def predict_with_mlp(model, class_names, scaler_mean, scaler_scale, feature_vector):
    feature_vector = (feature_vector - scaler_mean) / scaler_scale
    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_name, confidence


def load_dt_predictor(model_path):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    class_names = list(bundle["class_names"])
    return model, class_names


def predict_with_dt(model, class_names, feature_vector):
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    pred_idx = int(model.predict(x)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        confidence = float(probs[pred_idx])
    else:
        confidence = 1.0

    pred_name = class_names[pred_idx]
    return pred_name, confidence


def predict_with_selected_model(predictors, selected_model, feature_vector):
    if selected_model == "dt":
        model, class_names = predictors["dt"]
        return predict_with_dt(model, class_names, feature_vector)

    model, class_names, scaler_mean, scaler_scale = predictors["mlp"]
    return predict_with_mlp(model, class_names, scaler_mean, scaler_scale, feature_vector)


def choose_final_prediction(records, class_names):
    if not records:
        return None, 0.0, 0

    stats = {name: {"count": 0, "conf_sum": 0.0} for name in class_names}

    for pred_name, confidence in records:
        stats[pred_name]["count"] += 1
        stats[pred_name]["conf_sum"] += confidence

    best_name = None
    best_score = -1.0
    best_avg_conf = 0.0
    best_count = 0

    for name, values in stats.items():
        if values["count"] == 0:
            continue

        avg_conf = values["conf_sum"] / values["count"]
        score = values["count"] * avg_conf

        if score > best_score:
            best_name = name
            best_score = score
            best_avg_conf = avg_conf
            best_count = values["count"]

    return best_name, best_avg_conf, best_count


# ==============================
# DETECTOR AND IMAGE ASSETS
# ==============================

def make_hand_detector():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
    return HandLandmarker.create_from_options(options)


def load_icons(size=46):
    icons = {}
    for name in MUDRAS:
        path = os.path.join(REFERENCE_PATH, f"{name}.png")
        icon = cv2.imread(path)
        if icon is None:
            icons[name] = None
            continue
        icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
        icons[name] = icon
    return icons


ICONS = load_icons()


def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_centered(img, text, center_x, y, scale=0.9, color=WHITE, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    put_text(img, text, (int(center_x - tw / 2), y), scale, color, thickness, font)


def load_header_icon(size=58):
    icon = cv2.imread(HEADER_ICON_PATH, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None

    h, w = icon.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(icon, (new_w, new_h), interpolation=cv2.INTER_AREA)


def paste_header_icon(img, icon, center_x, center_y):
    if icon is None:
        return

    h, w = icon.shape[:2]
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return

    icon_crop = icon[y1 - y:y2 - y, x1 - x:x2 - x]
    rgb = icon_crop[:, :, :3]

    if icon_crop.shape[2] == 4:
        alpha = icon_crop[:, :, 3].astype(np.float32) / 255.0
    else:
        # Many character PNGs have a black background instead of transparency.
        alpha = (np.max(rgb, axis=2) > 12).astype(np.float32)

    alpha = alpha[:, :, None]
    roi = img[y1:y2, x1:x2]
    roi[:] = (rgb.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


HEADER_ICON = load_header_icon()


# ==============================
# DRAWING HELPERS
# ==============================

def draw_panel(img, x, y, w, h, title=None, number=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)
    cv2.rectangle(img, (x + 1, y + 1), (x + w - 1, y + h - 1), (4, 7, 10), 1)

    if title:
        if number is not None:
            put_text(img, f"{number}.", (x + 12, y + 21), 0.55, ORANGE, 2)
            put_text(img, title.upper(), (x + 36, y + 21), 0.47, WHITE, 1)
        else:
            put_text(img, title.upper(), (x + 14, y + 24), 0.52, WHITE, 1)


def fit_into_panel(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), (6, 8, 11), dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def paste_icon(img, name, x, y, size=46, checked=False):
    icon = ICONS.get(name)
    if icon is None:
        cv2.circle(img, (x + size // 2, y + size // 2), size // 2, BORDER, 1)
        put_centered(img, name[:2].upper(), x + size // 2, y + size // 2 + 5, 0.35, WHITE, 1)
    else:
        icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2 - 1, 255, -1)
        roi = img[y:y + size, x:x + size]
        if roi.shape[:2] == icon.shape[:2]:
            roi[mask > 0] = icon[mask > 0]

    cv2.circle(img, (x + size // 2, y + size // 2), size // 2, BORDER, 1)
    if checked:
        cv2.circle(img, (x + size - 6, y + 6), 7, GREEN, 1)
        cv2.line(img, (x + size - 9, y + 6), (x + size - 6, y + 9), GREEN, 2)
        cv2.line(img, (x + size - 6, y + 9), (x + size - 1, y + 2), GREEN, 2)


# ==============================
# DASHBOARD PANELS
# ==============================

def draw_header(img):
    cv2.rectangle(img, (0, 0), (WINDOW_W, 74), (5, 8, 11), -1)
    cv2.line(img, (0, 74), (WINDOW_W, 74), BORDER, 1)
    put_centered(img, "NARUTO", WINDOW_W // 2 - 190, 43, 1.1, ORANGE, 3, cv2.FONT_HERSHEY_TRIPLEX)
    put_centered(img, "HAND SIGN RECOGNITION", WINDOW_W // 2 + 85, 43, 1.05, WHITE, 2, cv2.FONT_HERSHEY_TRIPLEX)
    put_centered(img, "PERFORM A SEQUENCE, UNLEASH A JUTSU", WINDOW_W // 2, 65, 0.38, MUTED, 1)

    paste_header_icon(img, HEADER_ICON, 43, 37)
    paste_header_icon(img, HEADER_ICON, WINDOW_W - 43, 37)


def draw_webcam(img, frame):
    x, y, w, h = 8, 76, 450, 220
    draw_panel(img, x, y, w, h, "Live Webcam Feed", 1)
    view = fit_into_panel(frame, w - 14, h - 36)
    img[y + 29:y + 29 + view.shape[0], x + 7:x + 7 + view.shape[1]] = view
    cv2.circle(img, (x + 17, y + 47), 4, RED, -1)
    put_text(img, "REC", (x + 27, y + 51), 0.34, MUTED, 1)
    put_text(img, "FPS: 30", (x + w - 52, y + 51), 0.33, WHITE, 1)
    put_text(img, "1280x720", (x + 12, y + h - 13), 0.33, WHITE, 1)


def draw_detected_sign(img, sign, confidence=None):
    x, y, w, h = 8, 302, 450, 138
    draw_panel(img, x, y, w, h, "Detected Sign", 2)
    paste_icon(img, sign, x + 45, y + 43, 76)
    put_text(img, sign.upper(), (x + 162, y + 96), 0.78, GREEN, 2, cv2.FONT_HERSHEY_DUPLEX)

    if confidence is not None:
        put_text(
            img,
            f"Confidence: {confidence:.2f}",
            (x + 162, y + 123),
            0.45,
            MUTED,
            1,
        )


def draw_sequence_progress(img, current_sequence, target_sequence):
    x, y, w, h = 8, 448, 450, 144
    draw_panel(img, x, y, w, h, "Sequence Progress", 3)

    icon_y = y + 44
    start_x = x + 22
    gap = 55
    for i, name in enumerate(target_sequence):
        ix = start_x + i * gap
        done = i < len(current_sequence)
        paste_icon(img, name, ix, icon_y, 38, checked=done)
        put_centered(img, name.upper(), ix + 19, icon_y + 58, 0.32, GREEN if done else WHITE, 1)
        if i < len(target_sequence) - 1:
            put_text(img, ">", (ix + 42, icon_y + 25), 0.45, MUTED, 1)

    progress = min(len(current_sequence), len(target_sequence)) / max(1, len(target_sequence))
    bar_x, bar_y, bar_w, bar_h = x + 18, y + 118, w - 72, 12
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (28, 33, 39), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), GREEN, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), BORDER, 1)
    put_text(img, f"{len(current_sequence)} / {len(target_sequence)}", (bar_x + bar_w + 9, bar_y + 11), 0.42, GREEN, 1)


def draw_fireball_scene(img, frame, t, jutsu_name, unlocked=True):
    x, y, w, h = CENTER_PANEL
    draw_panel(img, x, y, w, h, f"Jutsu Active", 4)
    scene = fit_into_panel(frame, w - 10, h - 32)
    img[y + 29:y + h - 3, x + 5:x + w - 5] = scene


def draw_prediction_history(img, history):
    x, y, w, h = 1027, 76, 300, 188
    draw_panel(img, x, y, w, h, "Prediction History", 5)
    row_y = y + 52
    for item in history[:5]:
        timestamp, sign, ok = item
        put_text(img, timestamp, (x + 14, row_y), 0.4, MUTED, 1)
        put_text(img, sign.upper(), (x + 60, row_y), 0.44, WHITE, 1)
        if ok:
            cv2.rectangle(img, (x + w - 24, row_y - 12), (x + w - 12, row_y), GREEN, 1)
            cv2.line(img, (x + w - 22, row_y - 6), (x + w - 18, row_y - 2), GREEN, 1)
            cv2.line(img, (x + w - 18, row_y - 2), (x + w - 12, row_y - 11), GREEN, 1)
        row_y += 30


def draw_model_panel(img, selected_model):
    x, y, w, h = 1027, 270, 300, 184
    draw_panel(img, x, y, w, h, "Prediction Model", 6)

    models = [
        ("mlp", "MLP", "Multi-layer perceptron"),
        ("dt", "DT", "Decision tree"),
    ]
    row_y = y + 55

    for model_key, short_name, description in models:
        selected = selected_model == model_key
        color = GREEN if selected else MUTED
        border_color = GREEN if selected else BORDER

        cv2.rectangle(img, (x + 18, row_y - 24), (x + w - 18, row_y + 16), PANEL_2, -1)
        cv2.rectangle(img, (x + 18, row_y - 24), (x + w - 18, row_y + 16), border_color, 1)
        put_text(img, short_name, (x + 32, row_y + 2), 0.62, color, 2, cv2.FONT_HERSHEY_DUPLEX)
        put_text(img, description, (x + 91, row_y), 0.42, WHITE if selected else MUTED, 1)
        if selected:
            put_text(img, "", (x + w - 76, row_y), 0.36, GREEN, 1)
        row_y += 50

    put_centered(img, "M = MLP | D = DECISION TREE", x + w // 2, y + h - 18, 0.38, WHITE, 1)


def draw_status(img, status):
    x, y, w, h = 1027, 462, 300, 130
    draw_panel(img, x, y, w, h, "Status", 7)
    put_centered(img, "", x + w // 2, y + 64, 1.15, GREEN, 2, cv2.FONT_HERSHEY_COMPLEX)
    put_centered(img, status.upper(), x + w // 2, y + 93, 0.72, GREEN, 2, cv2.FONT_HERSHEY_DUPLEX)
    put_centered(img, "1-4 or F/L/W/S choose | SPACE starts", x + w // 2, y + 118, 0.38, WHITE, 1)


def draw_jutsu_card(img, x, y, w, h, jutsu, current_sequence, number, selected=False):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_2, -1)
    border_color = jutsu["color"] if selected else BORDER
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2 if selected else 1)
    put_text(img, str(number), (x + 10, y + 18), 0.42, MUTED, 1)
    put_text(img, jutsu["symbol"], (x + 17, y + 49), 1.3, jutsu["color"], 2, cv2.FONT_HERSHEY_DUPLEX)
    put_text(img, jutsu["name"], (x + 58, y + 20), 0.42, jutsu["color"], 1, cv2.FONT_HERSHEY_SIMPLEX)

    icon_x = x + 60
    icon_y = y + 29
    for i, sign in enumerate(jutsu["sequence"]):
        checked = i < len(current_sequence) and current_sequence[i] == sign
        paste_icon(img, sign, icon_x + i * 39, icon_y, 28, checked=checked)
        if i < len(jutsu["sequence"]) - 1:
            put_text(img, ">", (icon_x + i * 39 + 31, icon_y + 20), 0.32, MUTED, 1)


def draw_available_jutsus(img, current_sequence, target_jutsu):
    y = 598
    put_centered(img, "CHOOSE JUTSU: 1-4 OR FIRST LETTER", WINDOW_W // 2, y + 15, 0.48, WHITE, 1)
    card_w = 290
    card_h = 70
    start_x = 10
    for i, jutsu in enumerate(JUTSUS):
        selected = jutsu is target_jutsu
        draw_jutsu_card(
            img,
            start_x + i * (card_w + 10),
            y + 24,
            card_w,
            card_h,
            jutsu,
            current_sequence,
            i + 1,
            selected,
        )
    put_centered(img, "Built with OpenCV | MediaPipe | PyTorch | ScikitLearn | MLP + Decision Tree Models", WINDOW_W // 2, WINDOW_H - 8, 0.36, MUTED, 1)


# ==============================
# DASHBOARD COMPOSITION
# ==============================

# The filled values were just for the first test of the UI 
def create_dashboard(
    frame,
    detected_sign="ram",
    confidence=None,
    current_sequence=None,
    target_jutsu=None,
    status="Ready",
    history=None,
    selected_model="mlp",
):
    if target_jutsu is None:
        target_jutsu = JUTSUS[0]
    if current_sequence is None:
        current_sequence = ["boar", "ram", "tiger"]
    if history is None:
        history = [
            ("00:02", "boar", True),
            ("00:04", "ram", True),
            ("00:06", "tiger", True),
            ("00:08", "rat", True),
            ("00:10", "dog", True),
        ]

    dashboard = np.full((WINDOW_H, WINDOW_W, 3), BG, dtype=np.uint8)
    draw_header(dashboard)
    draw_webcam(dashboard, frame)
    draw_detected_sign(dashboard, detected_sign, confidence)
    draw_sequence_progress(dashboard, current_sequence, target_jutsu["sequence"])
    draw_fireball_scene(
        dashboard,
        frame,
        time.time(),
        target_jutsu["name"],
        unlocked=len(current_sequence) >= len(target_jutsu["sequence"]),
    )
    draw_prediction_history(dashboard, history)
    draw_model_panel(dashboard, selected_model)
    draw_status(dashboard, status)
    draw_available_jutsus(dashboard, current_sequence, target_jutsu)

    return dashboard


# ==============================
# MAIN WEBCAM LOOP
# ==============================

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"MLP checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(DT_MODEL_PATH):
        raise FileNotFoundError(f"Decision Tree model not found: {DT_MODEL_PATH}")

    print(f"Loading MediaPipe model: {MODEL_PATH}")
    detector = make_hand_detector()

    print(f"Loading MLP checkpoint: {CHECKPOINT_PATH}")
    mlp_predictor = load_mlp_predictor(CHECKPOINT_PATH)

    print(f"Loading Decision Tree model: {DT_MODEL_PATH}")
    dt_predictor = load_dt_predictor(DT_MODEL_PATH)

    predictors = {
        "mlp": mlp_predictor,
        "dt": dt_predictor,
    }
    selected_model = "mlp"
    class_names = predictors[selected_model][1]
    print(f"Classes: {class_names}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open webcam")
        detector.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    target = JUTSUS[0]
    _, _, effect_w, effect_h = CENTER_CONTENT
    jutsu_effects = make_jutsu_effects(effect_w, effect_h)
    active_effect_symbol = None
    current_sequence = []
    prediction_history = []
    detected_sign = "waiting"
    current_confidence = None
    status = "Press SPACE"
    frame_id = 0
    state = "idle"
    state_started_at = time.time()
    recorded_predictions = []
    result_correct = False

    print("Controls: 1-4 or F/L/W/S = choose jutsu | M = MLP | D = Decision Tree | SPACE/ENTER = start | R = reset | Q/ESC = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.flip(frame, 1)

        now = time.time()
        expected_idx = len(current_sequence)
        expected_sign = (
            target["sequence"][expected_idx]
            if expected_idx < len(target["sequence"])
            else None
        )

        if state == "idle":
            detected_sign = "waiting"
            current_confidence = None
            status = "Press SPACE"

        elif state == "prepare":
            elapsed = now - state_started_at
            remaining = max(0, int(math.ceil(PREPARE_SECONDS - elapsed)))
            detected_sign = "waiting"
            current_confidence = None
            status = f"Get {expected_sign} {remaining}s" if expected_sign else "Ready"

            if elapsed >= PREPARE_SECONDS:
                recorded_predictions = []
                detected_sign = "recording"
                current_confidence = None
                status = "Recording"
                state = "recording"
                state_started_at = now

        elif state == "recording":
            elapsed = now - state_started_at
            remaining = max(0, int(math.ceil(RECORD_SECONDS - elapsed)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_id / fps) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                feature_vector = extract_feature_vector(result)
                pred_name, confidence = predict_with_selected_model(
                    predictors,
                    selected_model,
                    feature_vector,
                )

                if confidence >= CONF_THRESHOLD:
                    recorded_predictions.append((pred_name, confidence))
                    detected_sign = pred_name
                    current_confidence = confidence
                else:
                    detected_sign = "waiting"
                    current_confidence = confidence
            else:
                detected_sign = "waiting"
                current_confidence = None

            status = f"Rec {remaining}s {len(recorded_predictions)}f"

            if elapsed >= RECORD_SECONDS:
                final_name, final_conf, final_count = choose_final_prediction(
                    recorded_predictions,
                    class_names,
                )

                has_enough_frames = final_count >= MIN_VALID_FRAMES
                has_enough_confidence = final_conf >= FINAL_CONF_THRESHOLD
                result_correct = (
                    has_enough_frames
                    and has_enough_confidence
                    and expected_sign is not None
                    and final_name == expected_sign
                )

                if has_enough_frames and has_enough_confidence and final_name is not None:
                    detected_sign = final_name
                    current_confidence = final_conf
                    elapsed_total = int(frame_id / fps)
                    prediction_history.insert(
                        0,
                        (f"00:{elapsed_total:02d}", final_name, result_correct),
                    )
                    prediction_history = prediction_history[:5]
                elif has_enough_frames and final_name is not None:
                    detected_sign = "not valid"
                    current_confidence = final_conf
                    result_correct = False
                else:
                    detected_sign = "not valid"
                    current_confidence = None
                    result_correct = False

                if result_correct:
                    current_sequence.append(final_name)
                    status = "Correct"
                elif has_enough_frames and not has_enough_confidence:
                    status = "Not valid"
                else:
                    status = "Try again"

                state = "result"
                state_started_at = now

        elif state == "result":
            if now - state_started_at >= RESULT_SECONDS:
                if len(current_sequence) >= len(target["sequence"]):
                    state = "complete"
                    detected_sign = "ready"
                    current_confidence = None
                    status = "Ready"
                    active_effect_symbol = target["symbol"]
                    trigger_jutsu_effect(jutsu_effects[active_effect_symbol], effect_w, effect_h)
                else:
                    state = "prepare"
                    state_started_at = now

        elif state == "complete":
            detected_sign = "ready"
            current_confidence = None
            status = "Ready"

        ui = create_dashboard(
            frame,
            detected_sign=detected_sign,
            confidence=current_confidence,
            current_sequence=current_sequence,
            target_jutsu=target,
            status=status,
            history=prediction_history,
            selected_model=selected_model,
        )

        effect_x, effect_y, effect_w, effect_h = CENTER_CONTENT
        effect_roi = ui[effect_y:effect_y + effect_h, effect_x:effect_x + effect_w]
        feed_jutsu_effects(jutsu_effects, effect_roi)
        if state == "complete" and active_effect_symbol is not None:
            effect_roi[:] = draw_jutsu_effect(jutsu_effects[active_effect_symbol], effect_roi)

        cv2.imshow("Naruto Hand Sign Recognition", ui)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        selected_idx = jutsu_index_from_key(key)
        if selected_idx is not None and state in ("idle", "complete"):
            target = JUTSUS[selected_idx]
            current_sequence = []
            prediction_history = []
            recorded_predictions = []
            detected_sign = "waiting"
            current_confidence = None
            status = f"Selected {target['name']}"
            state = "idle"
            active_effect_symbol = None
            state_started_at = time.time()
            reset_jutsu_effects(jutsu_effects)
        if key in (ord("m"), ord("M"), ord("d"), ord("D")) and state in ("idle", "complete"):
            selected_model = "mlp" if key in (ord("m"), ord("M")) else "dt"
            class_names = predictors[selected_model][1]
            prediction_history = []
            recorded_predictions = []
            detected_sign = "waiting"
            current_confidence = None
            status = f"Model {selected_model.upper()}"
            state = "idle"
            state_started_at = time.time()
        if key in (13, 32) and state in ("idle", "complete"):
            current_sequence = []
            prediction_history = []
            recorded_predictions = []
            detected_sign = "waiting"
            current_confidence = None
            status = "Prepare"
            state = "prepare"
            active_effect_symbol = None
            state_started_at = time.time()
            reset_jutsu_effects(jutsu_effects)
        if key == ord("r"):
            current_sequence = []
            prediction_history = []
            detected_sign = "waiting"
            current_confidence = None
            recorded_predictions = []
            status = "Press SPACE"
            state = "idle"
            active_effect_symbol = None
            state_started_at = time.time()
            reset_jutsu_effects(jutsu_effects)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
