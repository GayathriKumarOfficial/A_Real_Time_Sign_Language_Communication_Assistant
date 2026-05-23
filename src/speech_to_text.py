"""
speech_to_text.py
Records from PC microphone using PyAudio, saves clean WAV, transcribes with Whisper.
No ffmpeg needed.
"""

import pyaudio
import wave
import whisper
import threading
import os
import tempfile

# ── Audio settings ────────────────────────────────────────
CHUNK    = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000  # Whisper works best at 16kHz

# ── Global state ──────────────────────────────────────────
_whisper_model = None
_recording     = False
_frames        = []
_audio         = None
_stream        = None
_record_thread = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _get_input_device_index(pa):
    """Find first available input device. Returns None to use system default."""
    info = pa.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount', 0)
    for i in range(num_devices):
        dev = pa.get_device_info_by_host_api_device_index(0, i)
        if dev.get('maxInputChannels', 0) > 0:
            return i
    return None


def start_recording():
    """Start recording from the PC microphone in a background thread."""
    global _recording, _frames, _audio, _stream, _record_thread

    if _recording:
        return  # already running

    _frames    = []
    _recording = True

    try:
        _audio = pyaudio.PyAudio()
        device_index = _get_input_device_index(_audio)

        _stream = _audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,  # None = system default
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        _recording = False
        raise RuntimeError(f"Could not open microphone: {e}")

    def _capture():
        while _recording:
            try:
                data = _stream.read(CHUNK, exception_on_overflow=False)
                _frames.append(data)
            except Exception:
                break

    _record_thread = threading.Thread(target=_capture, daemon=True)
    _record_thread.start()


def stop_and_transcribe() -> str:
    """Stop recording, write WAV, transcribe with Whisper, return text."""
    global _recording, _stream, _audio, _record_thread

    if not _recording:
        raise RuntimeError("Not currently recording")

    # Stop the capture thread
    _recording = False
    if _record_thread:
        _record_thread.join(timeout=3)
        _record_thread = None

    # Close PyAudio stream
    try:
        if _stream:
            _stream.stop_stream()
            _stream.close()
            _stream = None
        if _audio:
            _audio.terminate()
            _audio = None
    except Exception:
        pass

    if not _frames:
        raise RuntimeError("No audio captured — is your microphone working?")

    # Write to a temporary WAV file
    tmp_path = tempfile.mktemp(suffix=".wav")
    try:
        pa_tmp = pyaudio.PyAudio()
        sample_width = pa_tmp.get_sample_size(FORMAT)
        pa_tmp.terminate()

        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(_frames))

        # Run Whisper
        model  = _load_whisper()
        result = model.transcribe(tmp_path, fp16=False, language='en')
        text   = result.get("text", "").strip()
        return text

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass