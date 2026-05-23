import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load
# --------------------------
dataset = np.load("landmarks/Hello/0.npy")
live = np.load("model/live_hello.npy")

print("Dataset:", dataset.shape)
print("Live:", live.shape)

# --------------------------
# Choose one landmark:
# Right hand wrist
# Left hand = first 63
# Right hand starts at 63
# wrist = first point of right hand
# x = index 63
# y = index 64
# --------------------------
ds_x = dataset[:, 63]
ds_y = dataset[:, 64]

lv_x = live[:, 63]
lv_y = live[:, 64]

# --------------------------
# Plot X movement
# --------------------------
plt.plot(ds_x, label="Dataset Hello")
plt.plot(lv_x, label="Live Hello")
plt.title("Right Wrist X Movement")
plt.xlabel("Frame")
plt.ylabel("X")
plt.legend()
plt.show()

# --------------------------
# Plot Y movement
# --------------------------
plt.plot(ds_y, label="Dataset Hello")
plt.plot(lv_y, label="Live Hello")
plt.title("Right Wrist Y Movement")
plt.xlabel("Frame")
plt.ylabel("Y")
plt.legend()
plt.show()