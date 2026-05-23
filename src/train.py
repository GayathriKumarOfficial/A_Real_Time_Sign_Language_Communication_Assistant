import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ------------------------------------
# Paths
# ------------------------------------
MODEL_PATH = "model"
OUTPUT_PATH = "outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ------------------------------------
# Load
# ------------------------------------
X = np.load(os.path.join(MODEL_PATH, "X.npy"))
y = np.load(os.path.join(MODEL_PATH, "y.npy"))

print("X:", X.shape)
print("y:", y.shape)

# ------------------------------------
# Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

# ------------------------------------
# Model
# Input = (30, 144)
# ------------------------------------
inputs = Input(shape=(X.shape[1], X.shape[2]))

x = Conv1D(
    filters=64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(inputs)

x = BatchNormalization()(x)

x = MaxPooling1D(pool_size=2)(x)

x = Dropout(0.3)(x)

x = Bidirectional(
    LSTM(
        128,
        return_sequences=True
    )
)(x)

x = Dropout(0.4)(x)

x = GlobalAveragePooling1D()(x)

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

outputs = Dense(
    y.shape[1],
    activation="softmax"
)(x)

model = Model(inputs, outputs)

# ------------------------------------
# Compile
# ------------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------
# Callbacks
# ------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# ------------------------------------
# Train
# ------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=80,
    batch_size=8,
    callbacks=[early_stop, reduce_lr]
)

# ------------------------------------
# Save model
# ------------------------------------
model.save(
    os.path.join(
        MODEL_PATH,
        "best_model.keras"
    )
)

# ------------------------------------
# Accuracy Curve
# ------------------------------------
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["Train", "Validation"])
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "accuracy_curve.png"
    )
)
plt.close()

# ------------------------------------
# Loss Curve
# ------------------------------------
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Train", "Validation"])
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "loss_curve.png"
    )
)
plt.close()

# ------------------------------------
# Confusion Matrix
# ------------------------------------
pred = model.predict(X_test)

pred = np.argmax(pred, axis=1)
true = np.argmax(y_test, axis=1)

cm = confusion_matrix(true, pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "confusion_matrix.png"
    )
)
plt.close()

# ------------------------------------
# Final score
# ------------------------------------
loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFINAL TEST ACCURACY:", acc)
print("Saved: model/best_model.keras")