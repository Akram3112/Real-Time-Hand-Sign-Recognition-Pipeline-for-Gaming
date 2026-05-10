    """_summary_ : This function is directly from mediapipe. 
    To show if the model is working and to visualize the landmarks.

    Returns:
        _type_: _description_ : See the original function in mediapipe for more details
    """



import cv2
import mediapipe as mp
import numpy as np

# ==============================
# MediaPipe Tasks API
# ==============================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandConnections = mp.tasks.vision.HandLandmarksConnections

# ==============================
# DRAW FUNCTION
# ==============================

def draw_landmarks_on_image(rgb_image, detection_result):

    annotated_image = np.copy(rgb_image)

    if detection_result.hand_landmarks is None:
        return annotated_image

    height, width, _ = annotated_image.shape

    for hand_landmarks in detection_result.hand_landmarks:

        # Draw points
        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(annotated_image,(x,y),5,(0,255,0),-1)

        # Draw connections
        for connection in HandConnections.HAND_CONNECTIONS:
        
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]

            x1 = int(start.x * width)
            y1 = int(start.y * height)

            x2 = int(end.x * width)
            y2 = int(end.y * height)

            cv2.line(annotated_image,(x1,y1),(x2,y2),(255,0,0),2)

    return annotated_image


# ==============================
# LOAD MODEL
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
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

frame_id = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, frame_id)
    frame_id += 1

    annotated = draw_landmarks_on_image(rgb, result)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    cv2.imshow("MediaPipe Hands", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()