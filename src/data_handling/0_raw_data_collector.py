import cv2
import os
import time

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = "1_dataset_VIDEO"
REFERENCE_PATH = "mudra_reference"
RECORD_TIME = 10  # seconds

mudra_labels = {
    1: "rat",
    2: "ox",
    3: "tiger",
    4: "hare",
    5: "dragon",
    6: "snake",
    7: "horse",
    8: "ram",
    9: "monkey",
    10: "bird",
    11: "dog",
    12: "boar"
}

# ==============================
# PERSON ID
# ==============================

person_id = int(input("Enter person ID (example: 1): "))
person_str = f"p{person_id:02d}"

# ==============================
# CREATE DATASET FOLDERS
# ==============================

for label in mudra_labels.values():
    os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera FPS:", fps)

# ==============================
# TAKE COUNTERS
# ==============================

take_counter = {label: 1 for label in mudra_labels.values()}

# ==============================
# KEY → MUDRA MAP
# ==============================

key_map = {
    ord('1'):1, ord('2'):2, ord('3'):3, ord('4'):4,
    ord('5'):5, ord('6'):6, ord('7'):7, ord('8'):8,
    ord('9'):9,
    ord('a'):10,
    ord('b'):11,
    ord('c'):12
}

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    cv2.putText(
        frame,
        "1-9 mudras | a=10 | b=11 | c=12 | q=quit",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.imshow("Naruto Mudra Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key not in key_map:
        continue

    mudra_id = key_map[key]
    label = mudra_labels[mudra_id]

    # ==============================
    # LOAD REFERENCE IMAGE
    # ==============================

    ref_path = os.path.join(REFERENCE_PATH, f"{label}.png")
    ref_img = None

    if os.path.exists(ref_path):
        ref_img = cv2.imread(ref_path)
        ref_img = cv2.resize(ref_img,(300,300))

    print(f"{label} selected. Press SPACE to start recording.")

    # ==============================
    # WAIT FOR SPACE
    # ==============================

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"{label} ready - press SPACE",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        if ref_img is not None:
            display_frame[10:310, width-310:width-10] = ref_img

        cv2.imshow("Naruto Mudra Recorder", display_frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord(' '):  # SPACE pressed
            break

        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ==============================
    # FILE NAME
    # ==============================

    take = take_counter[label]

    filename = f"{label}_{person_str}_t{take:02d}.mp4"
    video_path = os.path.join(DATASET_PATH,label,filename)

    take_counter[label] += 1

    # ==============================
    # VIDEO WRITER
    # ==============================

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path,fourcc,fps,(width,height))

    print("Recording:",filename)

    start_time = time.time()

    # ==============================
    # RECORD LOOP
    # ==============================

    while time.time() - start_time < RECORD_TIME:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)

        remaining = int(RECORD_TIME - (time.time()-start_time))

        cv2.putText(
            frame,
            f"{label} | {remaining}s",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        display_frame = frame.copy()

        if ref_img is not None:
            display_frame[10:310, width-310:width-10] = ref_img

        writer.write(frame)

        cv2.imshow("Naruto Mudra Recorder", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    print("Saved:",video_path)

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()