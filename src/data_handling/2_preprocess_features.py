import os
import pandas as pd
import numpy as np

RAW_FEATURES_PATH = "../../2_features"
PROCESSED_FEATURES_PATH = "../../3_features_processed"

os.makedirs(PROCESSED_FEATURES_PATH, exist_ok=True)

# Remove frames with too many zeros (usually missing hands / failed detection) need to play with it to find the best suited threshold
MAX_ZERO_RATIO = 0.65

# Remove near-duplicate consecutive frames
DUPLICATE_DISTANCE_THRESHOLD = 0.01

APPLY_SCALE_NORMALIZATION = False
APPLY_DEDUPLICATION = False
APPLY_MIRROR_AUGMENTATION = True

RELATIVE_COLUMNS = [
    "rel_wrist_dx_img",
    "rel_wrist_dy_img",
    "rel_wrist_dz_world",
    "rel_wrist_distance",
]

# World-landmark block size: 21 landmarks * 3 coords
HAND_BLOCK_SIZE = 63

# Angle features: 10 per hand
ANGLE_BLOCK_SIZE = 10

# Total features per hand: landmarks + angles = 73
TOTAL_HAND_FEATURES = HAND_BLOCK_SIZE + ANGLE_BLOCK_SIZE


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != "person_id"]



def get_world_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if c != "person_id" and c not in RELATIVE_COLUMNS and "_angle" not in c
    ]



def split_hand_columns(world_columns):
    handA_cols = [
        c for c in world_columns
        if (c.startswith("handA_") or c.startswith("left_")) and "_angle" not in c
    ]
    handB_cols = [
        c for c in world_columns
        if (c.startswith("handB_") or c.startswith("right_")) and "_angle" not in c
    ]
    return handA_cols, handB_cols


def has_two_hand_blocks(handA_cols, handB_cols) -> bool:
    return len(handA_cols) == HAND_BLOCK_SIZE and len(handB_cols) == HAND_BLOCK_SIZE



def zero_ratio(row_values: np.ndarray) -> float:
    if row_values.size == 0:
        return 1.0
    return float(np.mean(np.isclose(row_values, 0.0)))



def is_valid_row(row: pd.Series, feature_columns) -> bool:
    values = row[feature_columns].to_numpy(dtype=float)

    if zero_ratio(values) > MAX_ZERO_RATIO:
        return False

    if not np.isfinite(values).all():
        return False

    return True



def normalize_one_hand(values: np.ndarray) -> np.ndarray:
    """
    Normalize one hand block (63 values):
    1. Center all landmarks relative to the wrist (landmark 0), so the
       gesture is position-invariant regardless of where on screen the
       hand appears.
    2. Scale by the wrist-to-middle-fingertip distance (landmark 12),
       so the gesture is size-invariant regardless of hand distance to
       the camera.
    Landmark layout: [x1,y1,z1, x2,y2,z2, ...]
    """
    coords = values.reshape(21, 3).copy()

    wrist = coords[0].copy()
    coords -= wrist

    scale = np.linalg.norm(coords[12])
    if scale > 1e-8:
        coords /= scale

    return coords.reshape(-1)



def scale_normalize_row(row: pd.Series, handA_cols, handB_cols) -> pd.Series:
    row = row.copy()

    handA = row[handA_cols].to_numpy(dtype=float)
    handB = row[handB_cols].to_numpy(dtype=float)

    if not np.allclose(handA, 0.0):
        row.loc[handA_cols] = normalize_one_hand(handA)

    if not np.allclose(handB, 0.0):
        row.loc[handB_cols] = normalize_one_hand(handB)

    return row



def row_distance(row1: pd.Series, row2: pd.Series, feature_columns) -> float:
    v1 = row1[feature_columns].to_numpy(dtype=float)
    v2 = row2[feature_columns].to_numpy(dtype=float)
    return float(np.linalg.norm(v1 - v2))



def deduplicate_consecutive_frames(df: pd.DataFrame, feature_columns) -> pd.DataFrame:
    if df.empty:
        return df

    kept_rows = [0]

    for i in range(1, len(df)):
        d = row_distance(df.iloc[i], df.iloc[kept_rows[-1]], feature_columns)
        if d >= DUPLICATE_DISTANCE_THRESHOLD:
            kept_rows.append(i)

    return df.iloc[kept_rows].reset_index(drop=True)



def mirror_row(row: pd.Series, handA_cols, handB_cols) -> pd.Series:
    
    mirrored = row.copy()

    # Mirror landmark coordinates
    handA = row[handA_cols].to_numpy(dtype=float).reshape(21, 3)
    handB = row[handB_cols].to_numpy(dtype=float).reshape(21, 3)

    mirrored_A = handB.copy()
    mirrored_B = handA.copy()

    mirrored_A[:, 0] *= -1  # Flip X coordinate
    mirrored_B[:, 0] *= -1  # Flip X coordinate

    mirrored.loc[handA_cols] = mirrored_A.reshape(-1)
    mirrored.loc[handB_cols] = mirrored_B.reshape(-1)

    # Mirror angle columns (swap handA and handB angles)
    angle_cols_A = [c for c in row.index if c.startswith("handA_") and "_angle" in c]
    angle_cols_B = [c for c in row.index if c.startswith("handB_") and "_angle" in c]
    
    for col_A, col_B in zip(angle_cols_A, angle_cols_B):
        mirrored[col_A] = row[col_B]
        mirrored[col_B] = row[col_A]

    # Mirror relative features
    if "rel_wrist_dx_img" in mirrored.index:
        mirrored["rel_wrist_dx_img"] = -float(row["rel_wrist_dx_img"])
    if "rel_wrist_dy_img" in mirrored.index:
        mirrored["rel_wrist_dy_img"] = float(row["rel_wrist_dy_img"])
    if "rel_wrist_dz_world" in mirrored.index:
        mirrored["rel_wrist_dz_world"] = float(row["rel_wrist_dz_world"])
    if "rel_wrist_distance" in mirrored.index:
        mirrored["rel_wrist_distance"] = float(row["rel_wrist_distance"])

    return mirrored


for filename in os.listdir(RAW_FEATURES_PATH):
    if not filename.endswith(".csv"):
        continue

    label = os.path.splitext(filename)[0]
    input_path = os.path.join(RAW_FEATURES_PATH, filename)
    output_path = os.path.join(PROCESSED_FEATURES_PATH, filename)

    print(f"\nProcessing: {filename}")

    df = pd.read_csv(input_path)

    if df.empty:
        print("   File is empty -> skipping")
        continue

    feature_columns = get_feature_columns(df)
    world_columns = get_world_columns(df)
    handA_cols, handB_cols = split_hand_columns(world_columns)
    can_process_hands = has_two_hand_blocks(handA_cols, handB_cols)

    if not can_process_hands:
        print(
            f"   Hand columns not recognized for normalization/mirroring "
            f"(found {len(handA_cols)} + {len(handB_cols)}, expected 63 + 63)"
        )

    before = len(df)
    df = df[df.apply(lambda row: is_valid_row(row, feature_columns), axis=1)].reset_index(drop=True)
    print(f"   Removed bad frames: {before - len(df)}")

    if APPLY_SCALE_NORMALIZATION and can_process_hands and not df.empty:
        df = df.apply(lambda row: scale_normalize_row(row, handA_cols, handB_cols), axis=1)
        print("   Applied scale normalization")

    if APPLY_DEDUPLICATION and not df.empty:
        person_chunks = []
        for person_id, group in df.groupby("person_id", sort=False):
            clean_group = deduplicate_consecutive_frames(group.reset_index(drop=True), feature_columns)
            person_chunks.append(clean_group)
        df = pd.concat(person_chunks, ignore_index=True) if person_chunks else df.iloc[0:0]
        print("   Removed near-duplicate consecutive frames")

    if APPLY_MIRROR_AUGMENTATION and can_process_hands and not df.empty:
        mirrored_rows = [mirror_row(row, handA_cols, handB_cols) for _, row in df.iterrows()]
        mirrored_df = pd.DataFrame(mirrored_rows, columns=df.columns)
        mirrored_df["person_id"] = mirrored_df["person_id"].astype(str) + "_m"
        df = pd.concat([df, mirrored_df], ignore_index=True)
        print("   Added mirrored augmentation")

    df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path} ({len(df)} rows)")

print("\nPreprocessing finished.")
