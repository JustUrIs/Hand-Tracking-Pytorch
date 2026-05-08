import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions

MODEL_PATH = "model/hand_landmarker.task"

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

# Connections for drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def count_fingers(landmarks, handedness):
    fingers = []
    # Thumb: compare tip x vs IP joint x
    if handedness == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    # Four fingers: tip y above PIP y = extended
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return sum(fingers)


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def get_bounding_box(landmarks, frame_shape):
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs)) - 20)
    y1 = max(0, int(min(ys)) - 20)
    x2 = min(w, int(max(xs)) + 20)
    y2 = min(h, int(max(ys)) + 20)
    return x1, y1, x2, y2


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

capture = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as detector:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        for i, (hand_landmarks, handedness_list) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            label = handedness_list[0].category_name
            count = count_fingers(hand_landmarks, label)

            draw_hand(frame, hand_landmarks)

            x1, y1, x2, y2 = get_bounding_box(hand_landmarks, frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.putText(
                frame, f"{label}: {count} fingers",
                (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
            )

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
