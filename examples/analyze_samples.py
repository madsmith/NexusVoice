import wave
import numpy as np
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt

def analyze_wav_file(filepath):
    with wave.open(str(filepath), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        waveform = np.frombuffer(raw_data, dtype=dtype)

        if n_channels > 1:
            waveform = waveform.reshape(-1, n_channels)
            waveform = waveform.mean(axis=1)  # Mix to mono

        mean_volume = np.mean(np.abs(waveform))
        std_dev = np.std(waveform)

        return mean_volume, std_dev

def extract_delay_from_filename(filename):
    match = re.match(r"delay([+-]\d+)\.wav", filename)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Show plot of mean/stddev vs offset")
    args = parser.parse_args()

    sample_dir = Path("samples")
    wav_files = sorted(sample_dir.glob("*.wav"), key=lambda f: extract_delay_from_filename(f.name) or 999999)

    offsets = []
    means = []
    std_devs = []

    print(f"{'Filename':20} | {'Offset':>6} | {'Mean Volume':>12} | {'Std Dev':>12}")
    print("-" * 60)

    for filepath in wav_files:
        offset = extract_delay_from_filename(filepath.name)
        if offset is None:
            continue
        mean_volume, std_dev = analyze_wav_file(filepath)
        print(f"{filepath.name:20} | {offset:6} | {mean_volume:12.2f} | {std_dev:12.2f}")

        offsets.append(offset)
        means.append(mean_volume)
        std_devs.append(std_dev)

    if args.graph:

        plt.figure(figsize=(10, 5))
        plt.errorbar(offsets, means, yerr=std_devs, fmt='o', capsize=4, label='Mean ± StdDev')

        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title("Mean Volume with Standard Deviation Error Bars")
        plt.xlabel("Delay Offset (samples)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()