import os
import numpy as np
from tensorflow.keras.utils import to_categorical

LANDMARK_PATH = "landmarks"
MODEL_PATH = "model"

os.makedirs(MODEL_PATH, exist_ok=True)

X = []
y = []
labels = {}

# ---------------------------------
# Small augmentation on landmarks
# ---------------------------------
def augment_sequence(seq):
    augmented = seq.copy()

    # slight gaussian noise
    noise = np.random.normal(
        0,
        0.01,
        augmented.shape
    )

    augmented = augmented + noise

    return augmented


# ---------------------------------
# Load all .npy
# ---------------------------------
for idx, label in enumerate(
        sorted(os.listdir(LANDMARK_PATH))):

    label_path = os.path.join(
        LANDMARK_PATH,
        label
    )

    if not os.path.isdir(label_path):
        continue

    labels[label] = idx

    for file in os.listdir(label_path):

        if not file.endswith(".npy"):
            continue

        path = os.path.join(
            label_path,
            file
        )

        data = np.load(path)

        # original
        X.append(data)
        y.append(idx)

        # augmented
        X.append(
            augment_sequence(data)
        )
        y.append(idx)

# ---------------------------------
# Convert
# ---------------------------------
X = np.array(X)
y = to_categorical(y)

# Save
np.save(
    os.path.join(MODEL_PATH, "X.npy"),
    X
)

np.save(
    os.path.join(MODEL_PATH, "y.npy"),
    y
)

print("Preprocessing complete.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", labels)