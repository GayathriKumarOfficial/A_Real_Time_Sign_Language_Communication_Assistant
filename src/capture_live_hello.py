import cv2
import numpy as np
import mediapipe as mp

SEQUENCE_LENGTH = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# --------------------------
# Normalize
# --------------------------
def normalize_points(points):
    if np.all(points == 0):
        return points

    pts = points.reshape(-1, 3)

    center = np.mean(pts[:, :2], axis=0)
    pts[:, :2] -= center

    scale = np.max(np.abs(pts[:, :2]))
    if scale == 0:
        scale = 1

    pts[:, :2] /= scale

    return pts.flatten()


# --------------------------
# Extract
# --------------------------
def extract_keypoints(results):

    lh = np.array(
        [[p.x, p.y, p.z]
         for p in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array(
        [[p.x, p.y, p.z]
         for p in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    pose = np.array(
        [[p.x, p.y, p.z]
         for p in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    lh = normalize_points(lh)
    rh = normalize_points(rh)
    pose = normalize_points(pose)

    return np.concatenate([lh, rh, pose])


# --------------------------
# Capture
# --------------------------
cap = cv2.VideoCapture(0)
sequence = []
capturing = False

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if capturing:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            cv2.putText(
                frame,
                f"Capturing {len(sequence)}/{SEQUENCE_LENGTH}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

            if len(sequence) == SEQUENCE_LENGTH:
                np.save(
                    "model/live_hello.npy",
                    np.array(sequence)
                )
                print("Saved live sequence.")
                break

        cv2.putText(
            frame,
            "Press S = Capture Hello | Q = Quit",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.imshow("Capture Hello", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("s"):
            capturing = True
            sequence = []

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()