import pyaudio
import logging
import numpy as np
import threading
import queue
import time
from scipy.signal import stft, istft, chirp, spectrogram

logger = logging.getLogger(__name__)

from audio_utils import AudioData, PlaybackBuffer, save_recording
from debug import TimeThis

RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
MIC_BUFFER_CHUNKS = 32

class AudioDevice:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.mic_index = -1
        self.speaker_index = -1
        self._find_devices()
        
        self.mic_buffer = queue.deque(maxlen=MIC_BUFFER_CHUNKS)
        self.mic_buffer_unfiltered = queue.deque(maxlen=MIC_BUFFER_CHUNKS * 2)
        self.playback_queue = queue.Queue()
        self.playback_buffer = PlaybackBuffer(RATE)
        self.playback_buffer_lock = threading.Lock()
        self.mic_lock = threading.Lock()
        self.mic_buffer_ready = threading.Condition(self.mic_lock)
        self.mic_ready = threading.Event()
        self.playback_ready = threading.Event()
        self.read_in_progress = False
        self.read_did_overflow = False
        self.playback_is_playing = False
        self.playback_last_frame_time = 0

        self.running = True
        self.filtered = True
        self.delay = 0
        
        self.mic_warmup_frames = 2

        self._start_mic_thread()
        self._start_playback_thread()

        # Wait for both threads to be ready
        self.playback_ready.wait()
        self.mic_ready.wait()

    def set_delay(self, delay):
        self.delay = delay

    def set_sample_delay(self, delay):
        self.delay = delay / RATE

    def _find_devices(self):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if self.mic_index == -1 and info["maxInputChannels"] > 0:
                self.mic_index = i
            if self.speaker_index == -1 and info["maxOutputChannels"] > 0:
                self.speaker_index = i

    def _start_mic_thread(self):
        def mic_worker():
            with TimeThis("Mic thread init"):
                stream = self.audio.open(
                    format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                    input_device_index=self.mic_index)
            
            while self.running:
                mic_frame = AudioData(stream.read(CHUNK, exception_on_overflow=False))

                # Discard the first few frames as the microphone subsystem warms up
                if self.mic_warmup_frames > 0:
                    self.mic_warmup_frames -= 1
                    continue

                # Notify that the microphone is ready
                if not self.mic_ready.is_set():
                    self.mic_ready.set()

                filtered_frame = self._filter_frame(mic_frame)
                with self.mic_lock:
                    self.mic_buffer_unfiltered.append(mic_frame)
                    if len(self.mic_buffer) == self.mic_buffer.maxlen:
                        self.read_did_overflow = True
                    self.mic_buffer.append(filtered_frame.as_array())
                    self.mic_buffer_ready.notify_all()
            stream.stop_stream()
            stream.close()

        threading.Thread(target=mic_worker, daemon=True).start()

    def _start_playback_thread(self):
        def playback_worker():
            MAX_BUFFER_AGE = 3  # seconds

            with TimeThis("Playback thread init"):
                stream = self.audio.open(
                    format=FORMAT, channels=CHANNELS, rate=RATE,
                    output=True, output_device_index=self.speaker_index)

            while self.running:
                try:
                    # Notify that playback is ready
                    if not self.playback_ready.is_set():
                        self.playback_ready.set()
                    frames = self.playback_queue.get(timeout=0.1)

                    timestamp = time.perf_counter()
                    if timestamp - self.playback_last_frame_time < CHUNK / RATE:
                        timestamp = self.playback_last_frame_time

                    frame_data = AudioData(frames, timestamp=timestamp)
                    logger.debug(f"Playback Frame: {frame_data.timestamp:.3f} - {frame_data.end_time():.3f}")
                    self.playback_last_frame_time = frame_data.end_time()
                    with self.playback_buffer_lock:
                        self.playback_buffer.append(frame_data)
                        self.playback_buffer.prune_older_than(time.perf_counter() - MAX_BUFFER_AGE)
                    stream.write(frames)

                    time_to_next_frame = frame_data.end_time() - time.perf_counter()
                    sleep_time = max(0, time_to_next_frame - 0.01)
                    logger.debug(f"Playback Sleep: {sleep_time:.3f} sec")
                    time.sleep(sleep_time)
                except queue.Empty:
                    continue
            stream.stop_stream()

        threading.Thread(target=playback_worker, daemon=True).start()

    def _filter_frame(self, mic_frame: AudioData) -> AudioData:
        nperseg = CHUNK  # or another size you prefer
        noverlap = int(nperseg * 0.75)  # or another % you prefer

        delay_sec = self.delay

        frame_start = mic_frame.timestamp + delay_sec
        frame_end = frame_start + mic_frame.duration()

        with self.playback_buffer_lock:
            np_playback = self.playback_buffer.extract_frames(frame_start, len(mic_frame))

        # Print mic average volume
        mic_avg = np.mean(np.abs(mic_frame.as_array()))
        logger.debug(f"\nMic Frame: {mic_frame.timestamp:.3f} - {mic_frame.end_time():.3f} ({mic_avg:.1f})")

        if np.all(np_playback == 0):
            logger.debug(f"No playback found for this window")
            logger.debug(f"  Window: {frame_start:.3f} - {frame_end:.3f}")
            logger.debug(f"  Delay: {self.delay:.3f} sec")
            
            with self.playback_buffer_lock:
               str = self.playback_buffer.dump_windows("    ")
               logger.debug(str)
            return mic_frame
        
        
        # Convert to float32 for processing
        np_mic = mic_frame.as_array(np.float32)
        np_playback = np_playback.astype(np.float32)

        playback_avg = np.mean(np.abs(np_playback))
        logger.debug(f"Playback: {frame_start:.3f} - {frame_end:.3f} ({playback_avg:.1f})")

        # Ensure signals are same length
        min_length = min(len(np_playback), len(np_mic))
        if len(np_playback) < min_length:
            logger.info("Playback history is shorter than mic frame")
        if len(np_mic) < min_length:
            logger.info("Mic frame is shorter than playback history")
        np_mic = np_mic[:min_length]
        np_playback = np_playback[:min_length]

        if min_length < nperseg:
            logger.warning("Insufficient data for STFT", len(np_playback), len(np_mic))
            return mic_frame

        # Short-time Fourier Transform - Transform to frequency domain
        _, _, S_playback = stft(np_playback, fs=RATE, nperseg=nperseg, noverlap=noverlap)
        _, _, S_mic = stft(np_mic, fs=RATE, nperseg=nperseg, noverlap=noverlap)

        # Ensure signals are same length in frequency domain
        min_frames = min(S_playback.shape[1], S_mic.shape[1])
        S_playback = S_playback[:, :min_frames]
        S_mic = S_mic[:, :min_frames]

        # Subtracting spectral enegy of playback from microphone
        cleaned_magnitudes = np.maximum(np.abs(S_mic) - np.abs(S_playback), 0)
        cleaned_spectrum = cleaned_magnitudes * np.exp(1j * np.angle(S_mic))

        # Inverse Short-time Fourier Transform - Transform back to time domain
        _, cleaned_time = istft(cleaned_spectrum, fs=RATE, nperseg=nperseg, noverlap=noverlap)

        if np.any(np.isnan(cleaned_time)):
            logger.warning("NaNs in ISTFT output")
            return mic_frame

        # Clip the cleaned audio to the range of int16 and convert back to original format
        np_type = mic_frame.get_np_type()
        min_range = np.iinfo(np_type).min
        max_range = np.iinfo(np_type).max
        cleaned_clip = np.clip(cleaned_time, min_range, max_range).astype(np_type)

        return AudioData(cleaned_clip, format=mic_frame.format, channels=mic_frame.channels, rate=mic_frame.rate)

    def read(self):
        with self.mic_buffer_ready:
            # Wait for a frame to be available
            while len(self.mic_buffer) == 0:
                self.mic_buffer_ready.wait()

            if self.read_in_progress and not self.read_did_overflow:
                return self.mic_buffer.popleft()
            else:
                self.read_in_progress = True
                if self.read_did_overflow:
                    # TODO: alert the overflow to caller
                    self.read_did_overflow = False
                self.read_did_overflow = False
                last_chunk = self.mic_buffer.pop()
                self.mic_buffer.clear()
                return last_chunk

    def reset_microphone(self):
        with self.mic_lock:
            self.mic_buffer.clear()

    def play(self, audio_data):
        if isinstance(audio_data, AudioData):
            audio_data = audio_data.as_bytes()

        sample_size = pyaudio.get_sample_size(FORMAT)
        chunk_bytes = CHUNK * sample_size
        total_length = len(audio_data)
        
        chunks_queued = 0
        for i in range(0, total_length, chunk_bytes):
            chunk = audio_data[i:i + chunk_bytes]
            self.playback_queue.put(chunk)
            chunks_queued += 1

        logger.debug(f"Queued {chunks_queued} chunks for playback")

    def stop(self):
        self.running = False
        time.sleep(0.5)
        self.audio.terminate()



if __name__ == "__main__":
    VOLUME = 0.5

    class Note:
        def play(self, device):
            raise NotImplementedError
        
        def note(self, name: str, semitone_offset: int = 0) -> float:
            """
            Returns the frequency of a note with optional semitone adjustment.
            Example: note("C4") → 261.63
                    note("C4", 1) → C#4 ≈ 277.18
            """
            # Map note names to semitone offsets from C
            semitone_map = {
                'C': 0,  'C#': 1, 'Db': 1,
                'D': 2,  'D#': 3, 'Eb': 3,
                'E': 4,
                'F': 5,  'F#': 6, 'Gb': 6,
                'G': 7,  'G#': 8, 'Ab': 8,
                'A': 9,  'A#':10, 'Bb':10,
                'B':11
            }

            # Extract note name and octave
            name = name.strip()
            i = 1 if len(name) == 2 or name[1] not in "#b" else 2
            key = name[:i]
            
            # Default octave to 4 if not provided
            octave = int(name[i:]) if len(name) > i else 4

            # Calculate semitone distance from A4
            n = semitone_map[key] + (octave - 4) * 12 - 9 + semitone_offset

            # Frequency formula
            freq = 440.0 * (2 ** (n / 12))
            return round(freq, 2)
        
    class Rest(Note):
        def __init__(self, duration):
            self.duration = duration

        def play(self, device):
            time.sleep(self.duration)

    class Tone(Note):
        def __init__(self, freq, duration, fade=None):
            if isinstance(freq, str):
                freq = self.note(freq)
            self.freq = freq
            self.duration = duration
            self.fade = fade

        def play(self, device):
            device.play(self.render())
            time.sleep(self.duration)

        def render(self):
            if self.fade is None:
                self.fade = self.duration / 3
                self.fade = 0
            return self.generate_tone(self.freq, self.duration, self.fade)

        # Generate tone
        def generate_tone(self, freq, duration, end_fade=0.25):
            t = np.linspace(0, duration, int(RATE * duration), False)
            tone = np.sin(freq * t * 2 * np.pi)

            # Apply linear fade-out over the last `end_fade` seconds
            if end_fade > 0 and end_fade < duration:
                fade_samples = int(RATE * end_fade)
                fade = np.linspace(1.0, 0.0, fade_samples)
                tone[-fade_samples:] *= fade  # Apply fade to end of tone

            tone = (tone * (32767 * VOLUME)).astype(np.int16)
            return tone.tobytes()

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
            
    def main():
        import wave
        import numpy as np
        from scipy.signal import correlate
        import time

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
            audio = pyaudio.PyAudio()
            with wave.open(filename, 'rb') as wf:
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
        
        import matplotlib.pyplot as plt

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

        sample = load_wav("test_ai_response_short.wav")
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
        from pathlib import Path
        from audio_utils import save_recording  # Make sure this is imported

        base_delay = delay  # your calculated delay

        def test_delay(d):
            device.set_sample_delay(d)
            print(f"Testing delay: {d}")
            device.play(sample)

            recording = record(device, sample_data.duration() + 1)
            sign_char = "+" if d >= 0 else ""
            filename = Path(f"samples/delay{sign_char}{d}.wav")
            save_recording(recording, filename)
            time.sleep(0.5)
            device.reset_microphone()


        for offset in range(0, 500, 5):
            test_delay(offset)
            if offset > 0:
                test_delay(-offset)
            
        

        # def record_then_playback():
        #     while True:
        #         d = 2
        #         # Record 2s of audio
        #         print("Recording...")
        #         recorded = []
        #         for _ in range(d * RATE // CHUNK):
        #             recorded.append(device.read())
        #         recorded = np.concatenate(recorded)
        #         print("Recording complete")

        #         time.sleep(1)

        #         # Play back recorded audio
        #         print("Playing back recorded audio...")
        #         device.play(recorded)
        #         print("Playback complete")
        #         time.sleep(d)

        # def play_music_every(duration):
        #     while True:
        #         time.sleep(5)
        #         print("Playing music...")
        #         play_music(device)
        #         time.sleep(duration)

        # play_thread = threading.Thread(target=play_music_every, args=(14,), daemon=True)
        # listen_thread = threading.Thread(target=record_then_playback, daemon=True)

        # play_thread.start()
        # listen_thread.start()

        # play_thread.join()
        # listen_thread.join()


    main()