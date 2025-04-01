import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
                
import numpy as np
import pyaudio
from scipy.signal import chirp, correlate, spectrogram
import time
import wave

from audio.AudioDevice import AudioDevice
from audio.music import Tone, Rest
from audio.utils import AudioData
from utils.debug import TimeThis

RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
VOLUME = 0.5


def play_music(device):
    duration = 0.7
    rest = 0.1
    music = list()

    def add_tones(duration=duration, rest=rest):
        music.extend([
            Tone("D4", duration),
            Rest(rest),
            Tone("E4", duration),
            Rest(rest),
            Tone("C4", duration),
            Rest(rest),
            Tone("C3", duration),
            Rest(rest),
            Tone("G3", duration),
            Rest(rest),
            Rest(rest),
        ])

    add_tones(duration=duration, rest=rest)

    for tone in music:
        tone.play(device)

# Generate a centered impulse (click)
def generate_pulse(length=CHUNK):
    pulse = np.zeros(length, dtype=np.int16)
    pulse[length // 2] = 30000  # high amplitude click
    pulse[length // 2 + 4] = 30000  # high amplitude click
    pulse[length // 2 + 16] = 30000  # high amplitude click
    pulse[length // 2 + 64] = 30000  # high amplitude click
    pulse[length // 2 + 128] = 30000  # high amplitude click
    return pulse.tobytes()

def generate_chirp(duration=0.5, f0=100, f1=800, rate=16000, amplitude=0.4):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    signal *= (32767 * amplitude)

    # signal2 = chirp(t, f0=f1, f1=f0, t1=duration, method='linear')
    # signal2 *= (32767 * amplitude)
    # signal = np.concatenate((signal, signal2))
    return signal.astype(np.int16).tobytes()

def measure_playback_to_mic_delay(device, chunk_size=CHUNK, record_chunks=7, signal_duration=1, with_plot=False):
    """
    Measures playback-to-mic roundtrip delay in samples using an impulse pulse.
    
    Returns:
        delay_samples (int): estimated roundtrip delay from playback to mic
    """


    print("Generating and playing impulse pulse...")
    # playback_signal = generate_pulse(chunk_size * 2)
    playback_signal = generate_chirp(duration=1/4, f0=100, f1=5000)
    recorded = []

    # Play the pulse
    device.play(playback_signal)

    # Record several CHUNKs to capture response
    mic_reads = 0
    device.reset_microphone()
    for _ in range(record_chunks):
        frame = device.read()
        mic_reads += 1
        if frame is not None:
            recorded.append(frame)
        else:
            print("Warning: missed mic frame during recording.")
        time.sleep(chunk_size / RATE)

    print(f"Recorded {mic_reads} chunks")
    # Flatten and convert mic recording
    mic_audio = np.concatenate(recorded).astype(np.float32)

    # Convert the pulse to float32 for matching
    pulse_array = np.frombuffer(playback_signal, dtype=np.int16).astype(np.float32)

    # Cross-correlate to estimate lag
    corr = correlate(mic_audio, pulse_array, mode='full')
    midpoint = len(pulse_array) - 1
    lag = np.argmax(corr) - midpoint
    print(f"Estimated roundtrip delay: {lag} samples ({lag / RATE:.3f} sec)")
    # Score correlation
    peak = np.max(corr)
    mean = np.mean(np.abs(corr))
    score = peak / mean
    print(f"Correlation score: {score:.3f}")
    if with_plot:
        plot_mic_and_signal(mic_audio, pulse_array, corr)

    return lag, score

def measure_delay_from_buffers(device, frame_count=10, with_plot=False):
    """
    Measure playback-to-mic delay using actual AudioData chunks in the mic buffer
    and their aligned windows from the playback buffer.

    Returns:
        delay_samples (int), correlation_score (float)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate, spectrogram

    playback_signal = generate_chirp(duration=1/6, f0=100, f1=7000)
    print(f"Measure: Playing chirp signal... {len(playback_signal)//pyaudio.get_sample_size(FORMAT)} frames")
    device.play(playback_signal)

    with device.mic_lock:
        print("Measure: Clearing unfiltered mic buffer")
        device.mic_buffer_unfiltered.clear()

    # Sleep to allow response (10 chunks @ 16kHz with CHUNK=1024 ≈ 640ms)
    with TimeThis(f"Measure: Waiting for {frame_count} frames"):
        time.sleep((CHUNK * frame_count) / RATE)

    # Grab the next N chunks after the chirp
    with device.mic_lock:
        print(f"Measure: Unfiltered Mic Buffer Size: {len(device.mic_buffer_unfiltered)}")
        mic_chunks = list(device.mic_buffer_unfiltered)[:frame_count] 

    # Concatenate mic audio and get time window
    array_chunks = [chunk.as_array() for chunk in mic_chunks]
    mic_audio = np.concatenate(array_chunks).astype(np.float32)
    start_time = mic_chunks[0].timestamp - 0.3
    end_time = mic_chunks[-1].end_time()

    with device.playback_buffer_lock:
        playback_audio = device.playback_buffer.extract_frames(start_time, len(mic_audio))

    # Skip if playback was silent
    if np.all(playback_audio == 0):
        with device.playback_buffer_lock:
            print("No playback found for this window")
            print(f"Window: start={start_time:.3f}, end={end_time:.3f}")
            print("Found windows", len(device.playback_buffer))
            for chunk in device.playback_buffer.chunks:
                print(f"  {chunk.timestamp:.3f} - {chunk.end_time():.3f}")
            return None, None

    # Correlate
    corr = correlate(mic_audio, playback_audio, mode='full')
    lag = np.argmax(corr) - (len(playback_audio) - 1)
    print(f"Delay from buffers: {lag} samples ({lag / RATE:.3f} sec)")

    # Score correlation
    peak = np.max(corr)
    mean = np.mean(np.abs(corr))
    score = peak / mean
    print(f"Correlation score: {score:.3f}")

    if with_plot:
        fig, axs = plt.subplots(7, 1, figsize=(12, 10))

        axs[0].plot(playback_audio)
        axs[0].set_title("Extracted Playback Signal")

        axs[1].plot(mic_audio)
        axs[1].set_title("Mic Signal")

        # Pad to equal length
        max_len = max(len(mic_audio), len(playback_audio))
        mic_audio_padded = np.pad(mic_audio, (0, max_len - len(mic_audio)))
        playback_audio_padded = np.pad(playback_audio, (0, max_len - len(playback_audio)))

        f_mic, t_mic, Sxx_mic = spectrogram(mic_audio_padded, fs=RATE, nperseg=512, noverlap=384)
        f_play, t_play, Sxx_play = spectrogram(playback_audio_padded, fs=RATE, nperseg=512, noverlap=384)

        axs[2].pcolormesh(t_mic, f_mic, 10 * np.log10(Sxx_mic + 1e-10), shading='gouraud')
        axs[2].set_ylabel("Frequency [Hz]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_title("Spectrogram of Mic")

        axs[3].pcolormesh(t_play, f_play, 10 * np.log10(Sxx_play + 1e-10), shading='gouraud')
        axs[3].set_ylabel("Frequency [Hz]")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_title("Spectrogram of Playback")

        # Overlaid spectrograms
        axs[4].pcolormesh(t_mic, f_mic, 10 * np.log10(Sxx_mic + 1e-10), shading='gouraud', alpha=0.8)
        axs[4].pcolormesh(t_play, f_play, 10 * np.log10(Sxx_play + 1e-10), shading='gouraud', alpha=0.2)
        axs[4].set_ylabel("Frequency [Hz]")
        axs[4].set_xlabel("Time [s]")
        axs[4].set_title("Overlayed Spectrograms (Unaligned)")

        # Aligned mic or playback for overlay
        lag_samples = np.argmax(corr) - (len(playback_audio) - 1)

        # Shift playback to align with mic
        if lag_samples > 0:
            playback_aligned = np.pad(playback_audio, (lag_samples, 0))[:max_len]
        elif lag_samples < 0:
            playback_aligned = playback_audio[-lag_samples:]
            playback_aligned = np.pad(playback_aligned, (0, max_len - len(playback_aligned)))
        else:
            playback_aligned = playback_audio[:max_len]

        f_aligned_mic, t_aligned_mic, Sxx_aligned_mic = spectrogram(mic_audio_padded, fs=RATE, nperseg=512, noverlap=384)
        f_aligned_play, t_aligned_play, Sxx_aligned_play = spectrogram(playback_aligned, fs=RATE, nperseg=512, noverlap=384)
        axs[5].pcolormesh(t_aligned_mic, f_aligned_mic, 10 * np.log10(Sxx_aligned_mic + 1e-10), shading='gouraud', alpha=0.8)
        axs[5].pcolormesh(t_aligned_play, f_aligned_play, 10 * np.log10(Sxx_aligned_play + 1e-10), shading='gouraud', alpha=0.15)
        axs[5].set_ylabel("Frequency [Hz]")
        axs[5].set_xlabel("Time [s]")
        axs[5].set_title("Overlayed Spectrograms (Aligned)")

        axs[6].plot(corr)
        axs[6].axvline(np.argmax(corr), color='r', linestyle='--', label='Peak Lag')
        axs[6].set_title("Correlation")
        axs[6].legend()

        plt.tight_layout()
        plt.show()

    return lag, score

def load_wav(filename):
    if isinstance(filename, Path):
        filename = str(filename)

    audio = pyaudio.PyAudio()
    with wave.open(filename, 'rb') as wf:
        print (wf.getframerate())
        assert wf.getnchannels() == CHANNELS
        assert wf.getframerate() == RATE
        assert wf.getsampwidth() == audio.get_sample_size(FORMAT)
        frame_bytes = wf.readframes(wf.getnframes())
    audio.terminate()
    return frame_bytes

def record(device, duration):
    # Record 2s of audio
    print("Recording...")
    recorded = []
    from math import ceil
    for _ in range(ceil(duration * RATE / CHUNK)):
        recorded.append(device.read())
    recorded = np.concatenate(recorded).astype(np.int16).tobytes()
    print("Recording complete")
    return recorded

def plot_mic_and_signal(mic_audio, chirp_array, corr, rate=16000):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    # 1. Chirp waveform
    axs[0].plot(chirp_array)
    axs[0].set_title("Playback Chirp Signal")

    # 2. Mic recording waveform
    axs[1].plot(mic_audio)
    axs[1].set_title("Mic Recording (Waveform)")

    # 3. Spectrogram of mic
    f, t, Sxx = spectrogram(mic_audio, fs=rate, nperseg=512, noverlap=384)
    axs[2].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axs[2].set_ylabel("Frequency [Hz]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_title("Spectrogram of Mic Recording")

    # 4. Correlation
    axs[3].plot(corr, label="Cross-correlation")
    axs[3].axvline(np.argmax(corr), color='red', linestyle='--', label="Peak")
    axs[3].set_title("Correlation Result")
    axs[3].legend()

    plt.tight_layout()
    plt.show()

def main():

    
    import matplotlib.pyplot as plt


    sample = load_wav(BASE_DIR / "resources" / "test_ai_response_short.wav")
    sample_data = AudioData(sample)

    print(f"Sample: {len(sample)//pyaudio.get_sample_size(FORMAT)} frames")
    with TimeThis("AudioDevice init"):
        device = AudioDevice()
    
    # Measure playback-to-mic delay
    # delay, score = measure_playback_to_mic_delay(device, signal_duration=1, with_plot=True)
    # print("Score:", score)
    # print("Delay:", delay)
    # time.sleep(2)

    capture_frames = int(0.5 * RATE) // CHUNK
    capture_frames = 10
    delay, score = measure_delay_from_buffers(device, capture_frames, with_plot=False)
    print("Score:", score)
    print("Delay:", delay)
    # device.set_sample_delay(delay + 135)
    device.set_sample_delay(delay)

    beep = AudioData(Tone("A4", 0.25).render())
    print(f"Beep: {len(beep)//pyaudio.get_sample_size(FORMAT)} frames")
    device.play(beep)
    print(f"Recording beep...")
    audio = record(device, beep.duration() + 0.25)
    time.sleep(0.5)
    print("Playing back recording...")
    device.play(audio)

    time.sleep(AudioData(audio).duration())
    time.sleep(2)

    print("Playing sample...")
    device.play(sample)
    audio = record(device, sample_data.duration() + 1)

    print("Playback and recording complete")
    time.sleep(1)
    print("Playing recorded audio...")
    print(f"Audio: {len(audio)//pyaudio.get_sample_size(FORMAT)} frames")
    device.play(audio)
    time.sleep(AudioData(audio).duration() + 0.5)

    return

    time.sleep(5)
    from audio.utils import save_recording  # Make sure this is imported

    base_delay = delay  # your calculated delay

    def test_delay(delay, offset):
        new_delay = delay + offset
        device.set_sample_delay(new_delay)
        print(f"Testing delay: {offset}")
        device.play(sample)

        recording = record(device, sample_data.duration() + 1)
        sign_char = "+" if offset >= 0 else ""
        filename = Path(f"samples/delay{sign_char}{offset}.wav")
        save_recording(recording, filename)
        time.sleep(0.5)
        device.reset_microphone()

    base_delay = -2559
    for offset in range(275, 320, 1):
        test_delay(base_delay, offset)
        #if offset > 0:
        #    test_delay(base_delay, -offset)
        


if __name__ == "__main__":
    main()