import argparse
import logging
import numpy as np
import pyaudio
import random
import time
from scipy.signal import chirp, correlate, spectrogram
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil, floor
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

enable_graph = False

def generate_chirp(duration=0.5, f0=100, f1=800, rate=16000, amplitude=0.4):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    signal *= (32767 * amplitude)

    # signal2 = chirp(t, f0=f1, f1=f0, t1=duration, method='linear')
    # signal2 *= (32767 * amplitude)
    # signal = np.concatenate((signal, signal2))
    return signal.astype(np.int16).tobytes()

def measure_delay_from_buffers(device: AudioDevice, chunk=CHUNK, frame_count=10, overfetch=0.6, with_plot=False):
    """
    Measure playback-to-mic delay using actual AudioData chunks in the mic buffer
    and their aligned windows from the playback buffer.

    Returns:
        delay_samples (int), correlation_score (float)
    """

    playback_signal = generate_chirp(duration=1/6, f0=100, f1=7000)
    # print(f"Measure: Playing chirp signal... {len(playback_signal)//pyaudio.get_sample_size(FORMAT)} frames")
    device.play(playback_signal)

    with device.mic_lock:
        # print("Measure: Clearing unfiltered mic buffer")
        device.mic_buffer_unfiltered.clear()

    # Sleep to allow response (10 chunks @ 16kHz with CHUNK=1024 ≈ 640ms)
    # with TimeThis(f"Measure: Waiting for {frame_count} frames"):
    time.sleep((chunk * frame_count) / RATE)

    # Grab the next N chunks after the chirp
    with device.mic_lock:
        # print(f"Measure: Unfiltered Mic Buffer Size: {len(device.mic_buffer_unfiltered)}")
        mic_chunks = list(device.mic_buffer_unfiltered)[:frame_count] 

    # Concatenate mic audio and get time window
    array_chunks = [chunk.as_array() for chunk in mic_chunks]
    mic_audio = np.concatenate(array_chunks).astype(np.float32)
    start_time = mic_chunks[0].timestamp - overfetch / 2

    with device.playback_buffer_lock:
        playback_audio = device.playback_buffer.extract_frames(start_time, len(mic_audio) + int(overfetch * RATE))

    # Skip if playback was silent
    if np.all(playback_audio == 0):
        with device.playback_buffer_lock:
            print("No playback found for this window")
            windows = device.playback_buffer.dump_windows("    ")
            print(f"  Start: {mic_chunks[0].timestamp:.3f} End: {mic_chunks[-1].end_time():.3f} Duration: {mic_chunks[-1].end_time() - mic_chunks[0].timestamp:.3f}")
            return None, None

    # Correlate
    corr = correlate(mic_audio, playback_audio, mode='full')
    lag = np.argmax(corr) - (len(playback_audio) - 1)
    # print(f"Delay from buffers: {lag} samples ({lag / RATE:.3f} sec)")

    # Score correlation
    peak = np.max(corr)
    mean = np.mean(np.abs(corr))
    score = peak / mean
    # print(f"Correlation score: {score:.3f}")

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

def record(device, duration, chunk_size=CHUNK):
    # Record 2s of audio
    recorded = []
    for _ in range(ceil(duration * RATE / chunk_size)):
        recorded.append(device.read(chunk_size))
    recorded = np.concatenate(recorded).astype(np.int16).tobytes()
    return AudioData(recorded)

def measure_baseline(device, sample, sample_delay, chunk_size=CHUNK, measurements=3, all_data=None):
    """
    Play the sample and record the response, then measure the avg volume of
    the recorded audio.  Take n measurements and then average them.
    """
    device.set_sample_delay(sample_delay)
    device.reset_microphone()

    measurements_values = []
    for i in range(measurements):
        # print(f"Playing sample {i + 1}/{measurements}...")
        device.play(sample)
        recorded = record(device, sample.duration() + 0.25, chunk_size=chunk_size)

        data = recorded.as_array()
        mean_volume = np.sqrt(max(np.mean(data**2), 0))
        std_dev = np.std(data)
        # print(f"  Mean Volume: {mean_volume:.2f}, Std Dev: {std_dev:.2f}")
        measurements_values.append((mean_volume, std_dev))

        device.reset_microphone()

    # Calculate average volume and stddev
    mean_volume = np.mean([d[0] for d in measurements_values])
    std_dev = np.mean([d[1] for d in measurements_values])
    if all_data is not None:
        for mean, std_dev in measurements_values:
            all_data.append((sample_delay, mean, std_dev))

    # print(f"Average Volume: {mean_volume:.2f}, Std Dev: {std_dev:.2f}")

    return mean_volume, std_dev

def test_chunk_size(chunk_size, measurements=3):
    global enable_graph

    device = AudioDevice(chunk_size=chunk_size)

    # Load playback sample
    sample = AudioData.from_wave(ROOT_DIR / "examples" / "resources" / "test_ai_response_short.wav")

    capture_frames = 10
    delay, score = measure_delay_from_buffers(device, chunk=chunk_size, frame_count=capture_frames, with_plot=enable_graph)
    # print(f"Measured delay: {delay} samples ({delay / RATE:.3f} sec) with score {score:.3f}")
    device.set_sample_delay(delay)
    device.reset_microphone()

    volume, _ = measure_baseline(device, sample, delay, chunk_size=chunk_size, measurements=measurements)
    # print(f"Volume: {volume:.2f}")

    device.shutdown()

    return volume

def _get_top_candidates(all_data, candidate_count, with_data=False):
    """
    Get the top candidates with the lowest mean volume in the list of samples
    """
    if with_data:
        best_candidates = sorted(all_data.items(), key=lambda x: 0 if len(x[1]) == float("inf") else np.mean(x[1]))[:candidate_count]
    else:
        best_candidates = sorted(all_data.items(), key=lambda x: 0 if len(x[1]) == 0 else np.mean(x[1]))[:candidate_count]
    return best_candidates

def test_generation(all_data, candidate_count, survive_count):
    global enable_graph

    # all_data looks like {chunk_size: [volume_sample1, volume_sample2, ...]}

    # Get the best candidates with the lowest mean volume in the list of samples
    best_candidates = _get_top_candidates(all_data, candidate_count)

    print(f"  Best candidates: ")
    for chunk_size, samples in best_candidates:
        if len(samples) == 0:
            mean_volume = float("nan")
            std_dev = float("nan")
        else:        
            mean_volume = np.mean(samples)
            std_dev = np.std(samples)
        print(f"    {chunk_size:>4}: {mean_volume:.2f} ± {std_dev:.2f}")

    # Test the best candidates
    test_count = 1
    for chunk_size, samples in best_candidates:
        if len(samples) == 0:
            mean_volume = float("nan")
            std_dev = float("nan")
            measurements = 2
        else:        
            mean_volume = np.mean(samples)
            std_dev = np.std(samples)
            measurements = 1

        print(f"  Testing [{test_count}]: Chunk size {chunk_size} (Mean: {mean_volume:.2f} ± {std_dev:.2f} [{len(samples)} samples])...\r", end="")
        volume = test_chunk_size(chunk_size, measurements=measurements)
        all_data[chunk_size].append(volume)
        test_count += 1

def mutate_generation(all_data, base_chunk, base_spread, candidate_count, survive_count):
    """
    Mutate the best candidates around the top survivors so we have a new generation"
    of candidates to test."
    """

    # Get the survivors
    best_candidates = _get_top_candidates(all_data, survive_count, with_data=True)

    print(f"  Survivors: ")
    for chunk_size, samples in best_candidates:
        if len(samples) == 0:
            mean_volume = float("nan")
            std_dev = float("nan")
        else:        
            mean_volume = np.mean(samples)
            std_dev = np.std(samples)
        print(f"    {chunk_size:>4}: {mean_volume:.2f} ± {std_dev:.2f}")

    mu = np.mean([chunk_size for chunk_size, _ in best_candidates])
    sigma = np.std([chunk_size for chunk_size, _ in best_candidates])
    print(f"  Mutating with mu={mu:.2f} and sigma={sigma:.2f}")

    def is_valid(candidate):
        return candidate > 256 and candidate < 8192

    # Generate new candidates around the survivors
    for _ in range(candidate_count - survive_count):
        mutators = [
            # Randomize around the mean
            lambda: int(random.gauss(mu, sigma)),
            # Randomize around a random best candidate
            lambda: int(random.choice(best_candidates)[0] + random.gauss(0, sigma)),
            # Randomize around a the best candidate
            lambda: int(best_candidates[0][0] + random.gauss(0, sigma)),
            # Randomize around a random best candidate with a fixed offset
            lambda: int(random.gauss(base_chunk, base_spread))
        ]
        weights = [0.2, 0.4, 0.1, 0.2]

        candidate = random.choices(mutators, weights=weights)[0]()

        # Create a mutation
        if not is_valid(candidate):
            continue

        if candidate not in all_data:
            all_data[candidate] = []

def main():
    global enable_graph

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Show plot of audio waveforms")
    parser.add_argument("--chunk", type=int, default=1200, help="Initial Median chunk size")
    parser.add_argument("--spread", type=int, default=100, help="Spread of initial random samples")
    parser.add_argument("--survivors", type=int, default=5, help="Survival count")
    parser.add_argument("--candidates", type=int, default=10, help="Candidate count")
    args = parser.parse_args()
    enable_graph = args.graph
    
    all_data = {}

    # Generate initial random candidate samples
    while len(all_data) < args.candidates:
        candidate = int(random.gauss(args.chunk, args.spread))
        if candidate < 256:
            continue
        elif candidate > 8192:
            continue

        all_data[candidate] = []

    gen = 1
    while True:
        print(f"\nGeneration {gen}:")
        gen += 1

        # Test all candidates
        test_generation(all_data, args.candidates, args.survivors)
        mutate_generation(all_data, args.chunk, args.spread, args.candidates, args.survivors)


if __name__ == "__main__":
    main()