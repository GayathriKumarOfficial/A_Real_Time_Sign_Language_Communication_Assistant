import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ----------------------------
# Load model
# ----------------------------
model = load_model("model/best_model.keras")

LABELS = [
    "Hello",
    "How_are_you",
    "Thank_you",
    "help_me",
    "you_are_welcome"
]

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.50
UPPER_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ----------------------------
# Normalize
# ----------------------------
def normalize_points(points):
    if np.all(points == 0):
        return points

    pts = points.reshape(-1, 3)

    center = np.mean(pts[:, :2], axis=0)
    pts[:, :2] -= center

    scale = np.max(np.abs(pts[:, :2]))
    if scale == 0:
        scale = 1.0

    pts[:, :2] /= scale

    return pts.flatten()


# ----------------------------
# Extract
# Left hand + Right hand + Upper pose
# ----------------------------
def extract_keypoints(results):

    # left hand
    if results.left_hand_landmarks:
        lh = np.array([
            [p.x, p.y, p.z]
            for p in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # right hand
    if results.right_hand_landmarks:
        rh = np.array([
            [p.x, p.y, p.z]
            for p in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # upper pose
    if results.pose_landmarks:
        pose = np.array([
            [
                results.pose_landmarks.landmark[i].x,
                results.pose_landmarks.landmark[i].y,
                results.pose_landmarks.landmark[i].z
            ]
            for i in UPPER_POSE
        ]).flatten()
    else:
        pose = np.zeros(len(UPPER_POSE) * 3)

    lh = normalize_points(lh)
    rh = normalize_points(rh)
    pose = normalize_points(pose)

    return np.concatenate([lh, rh, pose])


# ----------------------------
# Main
# ----------------------------
def run_prediction():

    cap = cv2.VideoCapture(0)

    current_label = "Waiting..."
    confidence = 0.0

    state = "countdown"
    countdown_start = time.time()
    sequence = []
    result_time = None

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # ------------------------
            # Draw ONLY hand landmarks
            # ------------------------
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

            # ------------------------
            # COUNTDOWN
            # ------------------------
            if state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = max(0, 3 - int(elapsed))

                cv2.putText(
                    frame,
                    f"Start signing in: {remaining}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                if elapsed >= 3:
                    state = "capture"
                    sequence = []

            # ------------------------
            # CAPTURE
            # ------------------------
            elif state == "capture":

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                cv2.putText(
                    frame,
                    f"Capturing: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                if len(sequence) == SEQUENCE_LENGTH:

                    X = np.array(sequence)
                    X = np.expand_dims(X, axis=0)

                    pred = model.predict(X, verbose=0)[0]

                    class_id = np.argmax(pred)
                    confidence = float(pred[class_id])

                    top3 = np.argsort(pred)[-3:][::-1]

                    print("\nTop predictions:")
                    for i in top3:
                        print(
                            LABELS[i],
                            ":",
                            round(float(pred[i]), 3)
                        )

                    if confidence >= CONF_THRESHOLD:
                        current_label = LABELS[class_id]
                    else:
                        current_label = "Uncertain"

                    result_time = time.time()
                    state = "result"

            # ------------------------
            # SHOW RESULT
            # ------------------------
            elif state == "result":

                if time.time() - result_time >= 2:
                    state = "countdown"
                    countdown_start = time.time()

            # ------------------------
            # Prediction display
            # ------------------------
            cv2.putText(
                frame,
                f"{current_label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Close window (X) to stop",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "BiLSTM Sign Prediction",
                frame
            )

            # needed for refresh
            if cv2.waitKey(1) == 27:
                break

            # if window manually closed
            if cv2.getWindowProperty(
                "BiLSTM Sign Prediction",
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_prediction()