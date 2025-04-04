import argparse
import logging
import numpy as np
import pyaudio
import time
from scipy.signal import chirp, correlate, spectrogram
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
import sys

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from audio.AudioDevice import AudioDevice
from audio.utils import AudioData
from utils.debug import TimeThis

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - '%(message)s'")

AUDIO_FORMAT = pyaudio.paInt16
CHUNK = 1024
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1

def generate_chirp(duration=0.5, f0=100, f1=800, rate=16000, amplitude=0.4):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    signal *= (32767 * amplitude)

    # signal2 = chirp(t, f0=f1, f1=f0, t1=duration, method='linear')
    # signal2 *= (32767 * amplitude)
    # signal = np.concatenate((signal, signal2))
    return signal.astype(np.int16).tobytes()

def measure_delay_from_buffers(device, frame_count=10, with_plot=False):
    """
    Measure playback-to-mic delay using actual AudioData chunks in the mic buffer
    and their aligned windows from the playback buffer.

    Returns:
        delay_samples (int), correlation_score (float)
    """

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

def record(device, duration):
    # Record 2s of audio
    print("Recording...")
    recorded = []
    for _ in range(ceil(duration * RATE / CHUNK)):
        recorded.append(device.read())
    recorded = np.concatenate(recorded).astype(np.int16).tobytes()
    print("Recording complete")
    return AudioData(recorded)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Show plot of audio waveforms")
    args = parser.parse_args()

    # Initialize the AudioDevice
    device = AudioDevice(chunk_size=CHUNK)
    device.set_sample_delay(-2500)

    # Load playback sample
    sample = AudioData.from_wave(ROOT_DIR / "examples" / "resources" / "test_ai_response_short.wav")

    # capture_frames = 10
    # delay, score = measure_delay_from_buffers(device, capture_frames, with_plot=args.graph)
    # print("Score:", score)
    # print("Delay:", delay)

    # device.set_sample_delay(delay)
    # device.reset_microphone()

    print("Playing sample...")
    device.play(sample)
    playback_audio = record(device, sample.duration() + 0.25)

    time.sleep(1)
    print("Playing recorded audio...")
    device.play(playback_audio)

    # Dispaly statistics about volume of recorded audio
    data = playback_audio.as_array()
    volume = np.sqrt(np.max(np.mean(data**2), 0))
    peak_volume = np.max(np.abs(data))
    std_dev = np.std(data)
    print(f"Peak volume: {peak_volume:.2f}")
    print(f"Mean volume: {volume:.2f} ± {std_dev:.2f}")
    
    time.sleep(playback_audio.duration() + 0.25)


if __name__ == "__main__":
    main()