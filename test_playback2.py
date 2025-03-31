import numpy as np
import pyaudio
import threading
import time
import wave
from scipy.signal import correlate

# === Config ===
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

FILTER_LEN = 1024
MU = 1e-8
MAX_LAG = 512  # samples
MAX_ABS_VAL = 1e4

# === Setup ===
audio = pyaudio.PyAudio()
mic_device_index = -1
speaker_device_index = -1

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if mic_device_index == -1 and info["maxInputChannels"] > 0:
        mic_device_index = i
        print(f"Selected mic: {info['name']} (index {i})")
    if speaker_device_index == -1 and info["maxOutputChannels"] > 0:
        speaker_device_index = i
        print(f"Selected speaker: {info['name']} (index {i})")

# === Buffers ===
recorded_mic = []
played_audio = []

# === Utility ===
def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        assert wf.getnchannels() == CHANNELS
        assert wf.getframerate() == RATE
        assert wf.getsampwidth() == audio.get_sample_size(FORMAT)
        frames = wf.readframes(wf.getnframes())
    return frames

def save_wav(filename, data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data.astype(np.int16).tobytes())

def flatten_audio(buffers):
    return np.frombuffer(b"".join(buffers), dtype=np.int16)

def get_wav_duration(filename):
    with wave.open(filename, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()

# === Playback / Record Threads ===
def play_wav(filename):
    frames = load_wav(filename)
    played_audio.append(frames)

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        output_device_index=speaker_device_index)
    print(f"Playing: {filename}")
    stream.write(frames)
    stream.stop_stream()
    stream.close()

def record_mic(duration_seconds):
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=mic_device_index)

    frames_to_record = int(RATE / CHUNK * duration_seconds)
    print("Recording mic...")
    for _ in range(frames_to_record):
        data = stream.read(CHUNK)
        recorded_mic.append(data)

    stream.stop_stream()
    stream.close()
    print("Recording done.")

# === LMS Filter ===
def lms_filter(playback, mic, filter_len, mu):
    n = len(mic)
    w = np.zeros(filter_len, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)

    for i in range(filter_len, n):
        x = playback[i-filter_len:i][::-1]
        d = mic[i]
        y = np.dot(w, x)
        e = d - y
        output[i] = e

        # Stability check
        if np.any(np.isnan(x)) or np.isnan(e):
            continue
        if np.max(np.abs(x)) > MAX_ABS_VAL or abs(e) > MAX_ABS_VAL:
            continue

        w += 2 * mu * e * x

    return output

# === Main ===
filename = "input.wav"
total_duration = get_wav_duration(filename)

record_thread = threading.Thread(target=record_mic, args=(total_duration,))
play_thread = threading.Thread(target=play_wav, args=(filename,))

record_thread.start()
play_thread.start()
record_thread.join()
play_thread.join()

print(f"Captured {len(recorded_mic)} mic chunks and {len(played_audio)} playback segments.")

# === Postprocess ===
mic_signal = flatten_audio(recorded_mic)
playback_signal = flatten_audio(played_audio)

mic_f = mic_signal.astype(np.float32)
playback_f = playback_signal.astype(np.float32)

# === Constrained Cross-correlation ===
correlation = correlate(mic_f, playback_f, mode='full')
center = len(playback_f) - 1
lag_range = np.arange(center - MAX_LAG, center + MAX_LAG + 1)
restricted_corr = correlation[center - MAX_LAG : center + MAX_LAG + 1]
best_offset = np.argmax(restricted_corr)
lag = lag_range[best_offset] - center
print(f"Constrained lag: {lag} samples ({lag / RATE:.3f} sec)")

# === Alignment ===
if lag > 0:
    aligned_mic = mic_f[lag:]
    aligned_playback = playback_f[:len(aligned_mic)]
else:
    aligned_playback = playback_f[-lag:]
    aligned_mic = mic_f[:len(aligned_playback)]

final_len = min(len(aligned_mic), len(aligned_playback))
aligned_mic = aligned_mic[:final_len]
aligned_playback = aligned_playback[:final_len]

# === LMS Filtering ===
print("Running LMS...")
if final_len <= FILTER_LEN:
    print("Signal too short for LMS.")
    exit(1)

lms_output = lms_filter(aligned_playback, aligned_mic, FILTER_LEN, MU)
lms_output = lms_output.astype(np.int16)

# === Save Results ===
save_wav("test_mic_aligned.wav", aligned_mic)
save_wav("test_playback_aligned.wav", aligned_playback)
save_wav("test_cleaned_lms.wav", lms_output)

print("LMS cleaned WAV saved.")