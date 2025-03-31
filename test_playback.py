import numpy as np
import pyaudio
import threading
import time
from scipy.signal import stft, istft, correlate
import wave

# Audio configuration
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
VOLUME = 0.5

# PyAudio setup
audio = pyaudio.PyAudio()

# Device indices
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

# Buffers
recorded_mic = []
played_audio = []

# Generate tone
def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(RATE * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    tone = (tone * (32767 * VOLUME)).astype(np.int16)
    return tone.tobytes()

# Playback thread
def play_frequency_sequence(sequence):
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        output_device_index=speaker_device_index)

    for freq, duration in sequence:
        print(f"Playing {freq} Hz for {duration} sec")
        tone = generate_tone(freq, duration)
        played_audio.append(tone)
        stream.write(tone)

    stream.stop_stream()
    stream.close()

def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        assert wf.getnchannels() == CHANNELS
        assert wf.getframerate() == RATE
        assert wf.getsampwidth() == audio.get_sample_size(FORMAT)
        frames = wf.readframes(wf.getnframes())
    return frames

def play_wav(filename):
    frames = load_wav(filename)

    print(f"Playing: {filename}")
    play_frames(frames)

def play_frames(frames):
    played_audio.append(frames)  # for later subtraction
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        output_device_index=speaker_device_index)
    
    stream.write(frames)
    stream.stop_stream()
    stream.close()

# Recording thread
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

def get_wav_duration(filename):
    with wave.open(filename, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()
    
# Save audio to inspect results
def save_wav(filename, data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data.astype(np.int16).tobytes())

def generate_white_noise(duration_sec):
    samples = np.random.normal(0, 32767 * 0.5, int(RATE * duration_sec)).astype(np.int16)
    return samples.tobytes()

def lms_filter(playback, mic, filter_len=1024, mu=0.0000001):
    """
    Run LMS adaptive filtering.
    Args:
        playback (np.array): Known playback signal (float32)
        mic (np.array): Mic recording (float32)
        filter_len (int): Number of taps in the adaptive filter
        mu (float): Learning rate (careful, too high = unstable)
    Returns:
        np.array: Error signal (mic minus estimated echo)
    """
    n = len(mic)
    w = np.zeros(filter_len, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)

    MAX_ABS_VAL = 1e4  # prevent explosion

    for i in range(filter_len, n):
        x = playback[i-filter_len:i][::-1]  # reverse for FIR
        d = mic[i]
        y = np.dot(w, x)
        e = d - y
        output[i] = e

        # Clip before update
        if np.any(np.isnan(x)) or np.isnan(e):
            continue
        if np.max(np.abs(x)) > MAX_ABS_VAL or abs(e) > MAX_ABS_VAL:
            continue

        w += 2 * mu * e * x  # update weights
        w += (2 * mu * e * x) / (np.dot(x, x) + 1e-6)

    return output

def spectral_subtract(playback, mic, rate, n_fft=1024, overlap=0.75):
    hop_length = int(n_fft * (1 - overlap))

    # STFT
    f_p, t_p, Zx = stft(playback, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    f_m, t_m, Zy = stft(mic, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Make sure shapes match
    min_frames = min(Zx.shape[1], Zy.shape[1])
    Zx = Zx[:, :min_frames]
    Zy = Zy[:, :min_frames]

    # Magnitude subtraction (conservative)
    # cleaned_mag = np.maximum(np.abs(Zy) - 1 * np.abs(Zx), 0)
    mask = np.abs(Zx) / (np.abs(Zy) + 1e-6)
    suppression = np.clip(mask * 1.2, 0, 1.2)  # scale up
    cleaned_mag = np.maximum(np.abs(Zy) * (1 - suppression), 0)

    # mask = np.abs(Zx) / (np.abs(Zy) + 1e-6)
    # mask = np.clip(mask, 0, 1)
    # cleaned_mag = np.abs(Zy) * (1 - mask)

    # Phase from mic signal
    cleaned_spec = cleaned_mag * np.exp(1j * np.angle(Zy))

    # Inverse STFT
    _, cleaned_time = istft(cleaned_spec, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)

    return cleaned_time

# === MAIN ===
# Sequence to play
freq_sequence = [
    (440, 1.0),
    (660, 1.0),
    (880, 1.0),
]
freq_sequence = [
    (440, 0.2),
    (660, 0.2),
    (880, 0.2),
    (440, 0.2),
    (660, 0.2),
    (880, 0.2),
    (440, 0.2),
    (660, 0.2),
    (880, 0.2),
    (440, 0.2),
    (660, 0.2),
    (880, 0.2),
    (440, 0.2),
    (660, 0.2),
    (880, 0.2),
]

# # Generate frequency sequence
# freq_sequence = []

# # Create a generator that returns values between 0.01s and 0.1s and back to 0.01s repeating
# def duration_generator():
#     while True:
#         # Generate values oscillating between 0.01 and 0.1 in 0.01 increments and back to 0.01 forever
#         for i in range(1, 11):
#             yield i / 100
#         for i in range(9, 0, -1):
#             yield i / 100

# generator = duration_generator()
# for i in range(440, 880, 5):
#     # Take a duration from the generator
#     duration = next(generator)
#     freq_sequence.append((i, duration))


# Total playback duration
mode = "wavefile"

if mode == "tones":
    total_duration = sum(d for _, d in freq_sequence)
    play_thread = threading.Thread(target=play_frequency_sequence, args=(freq_sequence,))
elif mode == "wavefile":
    filename = "test_ai_response.wav"
    total_duration = get_wav_duration(filename)
    play_thread = threading.Thread(target=play_wav, args=(filename,))
elif mode == "noise":
    total_duration = 2.0
    noise = generate_white_noise(total_duration)
    play_thread = threading.Thread(target=play_frames, args=(noise,))
else:
    raise ValueError("Invalid mode")

# Start threads
record_thread = threading.Thread(target=record_mic, args=(total_duration,))

record_thread.start()
play_thread.start()

record_thread.join()
play_thread.join()

print(f"Captured {len(recorded_mic)} mic chunks and {len(played_audio)} tone segments.")

# Flatten buffers into one array each
def flatten_audio(buffers):
    return np.frombuffer(b"".join(buffers), dtype=np.int16)

mic_signal = flatten_audio(recorded_mic)
playback_signal = flatten_audio(played_audio)

# Convert to float32 for math
mic_f = mic_signal.astype(np.float32)
playback_f = playback_signal.astype(np.float32)

# === Cross-correlation to estimate lag ===
# correlation = correlate(mic_f, playback_f, mode='full')
# lag = np.argmax(correlation) - (len(playback_f) - 1)
# print(f"Estimated lag: {lag} samples ({lag / RATE:.3f} sec)")

MAX_LAG = 512  # about 32 ms at 16kHz

# Full correlation
correlation = correlate(mic_f, playback_f, mode='full')

# Center index
center = len(playback_f) - 1

# Focus only on plausible lags
lag_range = np.arange(center - MAX_LAG, center + MAX_LAG + 1)
restricted_corr = correlation[center - MAX_LAG : center + MAX_LAG + 1]

# Find best lag
best_offset = np.argmax(restricted_corr)
lag = lag_range[best_offset] - center

print(f"Constrained lag: {lag} samples ({lag / RATE:.3f} sec)")

# Align signals based on lag
if lag > 0:
    aligned_mic = mic_f[lag:]
    aligned_playback = playback_f[:len(aligned_mic)]
else:
    aligned_playback = playback_f[-lag:]
    aligned_mic = mic_f[:len(aligned_playback)]

# Final safety trim to equal length
final_len = min(len(aligned_mic), len(aligned_playback))
aligned_mic = aligned_mic[:final_len]
aligned_playback = aligned_playback[:final_len]


# === Estimate gain (alpha) and subtract ===
alpha = np.dot(aligned_mic, aligned_playback) / np.dot(aligned_playback, aligned_playback)
print(f"Estimated alpha: {alpha:.4f}")

# Apply LMS filter
if len(aligned_playback) < 1024:
    print("Playback too short for LMS filter length.")
    exit(1)

#print("Running LMS filter...")
#lms_output = lms_filter(aligned_playback, aligned_mic, filter_len=2048, mu=1e-9)
#lms_output = lms_output.astype(np.int16)

print("Running frequency-domain subtraction...")
fd_output = spectral_subtract(aligned_playback, aligned_mic, RATE, n_fft=2048*5, overlap=0.2)
fd_output = np.clip(fd_output, -32768, 32767).astype(np.int16)

save_wav("test_cleaned_fd.wav", fd_output)

# scaled_playback = aligned_playback * alpha
# cleaned_signal = aligned_mic - scaled_playback
# cleaned_signal = cleaned_signal.astype(np.int16)

# if np.any(np.isnan(lms_output)):
#     print("LMS output contains NaNs!")
# else:
#     save_wav("test_cleaned_lms.wav", lms_output.astype(np.int16))
#     print("LMS cleaned WAV saved.")

def save_stereo_wav(filename, left, right):
    # Ensure both are int16 and same length
    min_len = min(len(left), len(right))
    left = left[:min_len].astype(np.int16)
    right = right[:min_len].astype(np.int16)

    # Interleave as L-R-L-R...
    stereo = np.empty(2 * min_len, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(stereo.tobytes())

# Save stereo aligned comparison
save_stereo_wav("test_aligned_mic_vs_playback.wav", aligned_mic, aligned_playback)

save_wav("test_mic_aligned.wav", aligned_mic)
save_wav("test_playback_aligned.wav", aligned_playback)
#save_wav("test_cleaned_aligned.wav", cleaned_signal)
time.sleep(1)

play_wav("test_playback_aligned.wav")
play_wav("test_mic_aligned.wav")
play_wav("test_cleaned_fd.wav")