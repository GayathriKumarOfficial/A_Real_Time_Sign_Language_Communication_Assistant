import numpy as np
import os
from tensorflow.keras.models import load_model

LABELS = ["Hello", "How_are_you", "Thank_you", "help_me", "you_are_welcome"]
SEQ_LEN  = 30
FEATURES = 144
_model   = None

def _load():
    global _model
    if _model is None:
        # Try multiple path strategies
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, '..', 'model', 'best_model.keras'),
            os.path.join(base, 'model', 'best_model.keras'),
            'model/best_model.keras',
            'E:/A_Real_Time_ISL/model/best_model.keras',
        ]
        path = None
        for c in candidates:
            p = os.path.abspath(c)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"best_model.keras not found. Tried: {[os.path.abspath(c) for c in candidates]}"
            )
        print(f"[model] Loading from: {path}")
        _model = load_model(path)
        _model.predict(np.zeros((1, SEQ_LEN, FEATURES), dtype=np.float32), verbose=0)
        print("[model] Ready.")
    return _model

def normalize_points(flat):
    pts = np.array(flat, dtype=np.float32).reshape(-1, 3)
    if np.all(pts == 0):
        return pts.flatten()
    center = np.mean(pts[:, :2], axis=0)
    pts[:, :2] -= center
    scale = np.max(np.abs(pts[:, :2]))
    if scale == 0:
        scale = 1.0
    pts[:, :2] /= scale
    return pts.flatten()

def predict_from_sequence(sequence):
    model = _load()
    arr = np.array(sequence, dtype=np.float32)
    if arr.shape[0] != SEQ_LEN:
        idx = np.linspace(0, arr.shape[0]-1, SEQ_LEN).astype(int)
        arr = arr[idx]
    X     = arr[np.newaxis, ...]
    preds = model.predict(X, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return LABELS[idx], round(float(preds[idx]) * 100, 1)