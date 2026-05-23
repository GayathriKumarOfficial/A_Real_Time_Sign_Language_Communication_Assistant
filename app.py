from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'signlang_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CHAT_ROOM = "shared_room"

# load model once at startup
from src.predict_sign import predict_from_sequence
import numpy as np
try:
    predict_from_sequence([list(np.zeros(144))] * 30)
    print("[model] warmed up OK")
except Exception as e:
    print(f"[model] warm-up note: {e}")

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/deaf')
def deaf():
    return render_template('deaf_dashboard.html')

@app.route('/hearing')
def hearing():
    return render_template('hearing_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json(force=True)
        sequence = data.get('sequence')
        if not sequence or len(sequence) < 10:
            return jsonify({'error': 'Need at least 10 frames'}), 400
        label, confidence = predict_from_sequence(sequence)
        return jsonify({'label': label, 'confidence': confidence})
    except Exception as e:
        print(f"[predict error] {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def on_connect():
    join_room(CHAT_ROOM)

@socketio.on('join')
def on_join(data):
    join_room(CHAT_ROOM)
    role = data.get('role', 'User')
    emit('system_message', {
        'text': f'{role} has joined the session.',
        'timestamp': datetime.now().strftime('%H:%M')
    }, room=CHAT_ROOM)

@socketio.on('send_message')
def on_send_message(data):
    emit('receive_message', {
        'text':      data.get('text', ''),
        'sender':    data.get('sender', 'User'),
        'role':      data.get('role', 'deaf'),
        'timestamp': datetime.now().strftime('%H:%M'),
        'sid':       request.sid
    }, room=CHAT_ROOM)

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)