# SignBridge — Real-Time Sign Language Communication System

**B.Tech Final Year Project**  
Real-time two-way communication between Deaf users (signing) and Hearing users (voice/text).

[![Demo Video](https://img.shields.io/badge/▶%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/2PwIoqBvASs)

---

## 👥 Team

**Developed by:** [Gayathri K](https://github.com/GayathriKumarOfficial) & [Monika A D](https://github.com/Monikamaheswari04)  
**Guide:** Ms. S. Dhivya

---

## Project Overview

| Role | Input | Output |
|------|-------|--------|
| Deaf User | Hand signs via webcam | Predicted sign text → shared chat |
| Hearing User | Voice or typed text | Transcribed / typed text → shared chat |

**Model:** CNN + BiLSTM trained on custom webcam-close ISL dataset  
**Signs:** Hello, How are you, Thank you, Help me, You are welcome

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, Flask, Flask-SocketIO |
| ML Model | TensorFlow / Keras (CNN + BiLSTM) |
| Landmark Detection | MediaPipe Holistic (browser CDN) |
| Speech-to-Text | Web Speech API (browser-native) |
| Frontend | HTML, CSS, JavaScript, Socket.IO |
| Real-time | WebSocket via Flask-SocketIO |

---

## Project Structure

```
A_Real_Time_ISL/
├── app.py                        # Flask server + SocketIO + /predict route
├── requirements.txt
├── README.md
│
├── model/
│   └── best_model.keras          # Trained CNN+BiLSTM model (input: 30×144)
│
├── src/
│   ├── predict_sign.py           # Server-side Keras inference
│   ├── speech_to_text.py         # (legacy — replaced by Web Speech API)
│   ├── extract_landmarks.py      # Landmark extraction utility
│   ├── preprocess.py             # Data preprocessing
│   ├── train.py                  # Model training script
│   └── capture_live_hello.py     # Data capture utility
│
├── templates/
│   ├── login.html                # Role selection page
│   ├── deaf_dashboard.html       # Deaf user — signing + chat
│   └── hearing_dashboard.html    # Hearing user — voice/text + chat
│
├── static/
│   ├── style.css                 # Full UI stylesheet
│   └── chat.js                   # Shared chat bubble renderer
│
├── dataset/                      # Raw training videos
├── landmarks/                    # Extracted landmark .npy files
└── outputs/                      # Training outputs (plots, confusion matrix)
```

---

## Installation

### 1. Open the project
```bash
cd E:\A_Real_Time_ISL
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

Follow these steps in order:

### Step 1 — Extract Landmarks
Processes raw videos from `dataset/` and extracts MediaPipe hand + pose landmarks.
```bash
python src/extract_landmarks.py
```
Output saved to `landmarks/`

### Step 2 — Preprocess
Converts extracted landmarks into numpy sequences ready for training.
```bash
python src/preprocess.py
```
Output saved to `model/X.npy` and `model/y.npy`

### Step 3 — Train Model
Trains the CNN + BiLSTM model on the preprocessed sequences.
```bash
python src/train.py
```
Trained model saved to `model/best_model.keras`

### Step 4 — Run the Web App
Starts the Flask server with real-time sign prediction and chat.
```bash
python app.py
```

Server starts at: **http://127.0.0.1:5000**

> **Note:** Steps 1–3 only need to be run once. After `best_model.keras` is generated, you only need Step 4 to use the app.

---

## Usage

Open **two browser tabs** (Chrome or Edge recommended):

| Tab | URL | Role |
|-----|-----|------|
| Tab 1 | http://127.0.0.1:5000 → Deaf User | Signs via webcam |
| Tab 2 | http://127.0.0.1:5000 → Hearing User | Speaks or types |

Both tabs share a **single SocketIO room** — messages appear in real time on both sides.

---

## How It Works

### Deaf User Flow
```
Webcam
  → Browser MediaPipe Holistic (CDN)
  → Extract landmarks (left hand + right hand + upper pose)
  → normalize_points() per segment
  → Rolling 30-frame buffer (144 features/frame)
  → POST /predict  {sequence: [[144 floats] × 30]}
  → Flask → best_model.keras.predict()
  → Return {label, confidence}
  → Display on screen → Send to chat
```

### Hearing User Flow
```
Microphone
  → Browser Web Speech API (instant, no server needed)
  → Interim transcript shown live
  → Final transcript on Stop
  → Send to shared chat via SocketIO
```

### Chat Flow
```
Both users → SocketIO shared room → receive_message event → chat bubbles
```

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | CNN + BiLSTM |
| Input shape | (30, 144) |
| Output | Softmax over 5 classes |
| Saved as | `model/best_model.keras` |

### Feature Vector (144 floats per frame)

| Segment | Landmarks | Features |
|---------|-----------|----------|
| Left hand | 21 × xyz | 63 |
| Right hand | 21 × xyz | 63 |
| Upper pose (shoulders, elbows, wrists) | 6 × xyz | 18 |
| **Total** | | **144** |

### Normalization (per segment, matches training)
```python
center = mean(pts[:, :2], axis=0)   # mean of x, y
pts[:, :2] -= center                 # center subtract
scale = max(abs(pts[:, :2]))        # max abs of x, y
pts[:, :2] /= scale                 # scale divide
```

### Signs / Labels
```python
LABELS = ["Hello", "How_are_you", "Thank_you", "help_me", "you_are_welcome"]
```

---

## API Reference

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Login page |
| GET | `/deaf` | Deaf user dashboard |
| GET | `/hearing` | Hearing user dashboard |
| POST | `/predict` | Run sign prediction |

**POST `/predict`**
```json
Request:  { "sequence": [[144 floats], [144 floats], ...] }  // 30 frames
Response: { "label": "Hello", "confidence": 91.3 }
```

### SocketIO Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `join` | client → server | `{ "role": "Deaf User" }` |
| `send_message` | client → server | `{ "text", "sender", "role" }` |
| `receive_message` | server → all | `{ "text", "sender", "role", "timestamp" }` |
| `system_message` | server → all | `{ "text", "timestamp" }` |

---

## Dataset

- **Classes:** 5 signs
- **Videos:** 76 total (16 × Hello, 15 × each other class)
- **Recording setup:** Sitting close to laptop, upper body visible, front-facing webcam
- **Sequence length:** 30 frames per sample

---

## Demo

▶ Watch the project demo: [https://youtu.be/2PwIoqBvASs](https://youtu.be/2PwIoqBvASs)

---

## Known Requirements

- **Browser:** Chrome or Edge (required for Web Speech API on hearing side)
- **Python:** 3.10
- **MediaPipe in browser:** Loaded via CDN — requires internet on first load (cached after)
- **Model file:** `model/best_model.keras` must exist before running `app.py`

---

## Acknowledgements

Special thanks to our project guide **Ms. S. Dhivya** for her continuous support and guidance throughout this project.
