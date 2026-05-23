import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

DATASET_PATH = "dataset"
SAVE_PATH = "landmarks"

SEQUENCE_LENGTH = 30
VALID_EXT = [".mp4", ".mov", ".avi"]

UPPER_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic


# -----------------------------------
# Normalize
# -----------------------------------
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


# -----------------------------------
# Extract
# Hands + Upper Pose only
# -----------------------------------
def extract_keypoints(results):

    # left hand
    lh = np.array(
        [[p.x, p.y, p.z]
         for p in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # right hand
    rh = np.array(
        [[p.x, p.y, p.z]
         for p in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # upper pose only
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


# -----------------------------------
# Main
# -----------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    for label in os.listdir(DATASET_PATH):

        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        print(f"\nProcessing: {label}")

        save_label = os.path.join(SAVE_PATH, label)
        os.makedirs(save_label, exist_ok=True)

        videos = os.listdir(label_path)

        for idx, video in enumerate(tqdm(videos)):

            ext = os.path.splitext(video)[1].lower()

            if ext not in VALID_EXT:
                continue

            cap = cv2.VideoCapture(
                os.path.join(label_path, video)
            )

            frames = []

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = holistic.process(rgb)

                keypoints = extract_keypoints(results)
                frames.append(keypoints)

            cap.release()

            if len(frames) == 0:
                continue

            # fixed 30 frames
            if len(frames) >= SEQUENCE_LENGTH:
                step = len(frames) // SEQUENCE_LENGTH
                frames = frames[::step][:SEQUENCE_LENGTH]
            else:
                while len(frames) < SEQUENCE_LENGTH:
                    frames.append(
                        np.zeros_like(frames[0])
                    )

            np.save(
                os.path.join(save_label, f"{idx}.npy"),
                np.array(frames)
            )

print("\nExtraction complete.")