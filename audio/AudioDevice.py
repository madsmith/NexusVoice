import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

import pyaudio
import logging
from math import ceil
import numpy as np
import threading
import queue
import time
from scipy.signal import stft, istft

logger = logging.getLogger(__name__)

from audio.utils import AudioData, PlaybackBuffer, save_recording
from utils.debug import TimeThis

RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

MIC_BUFFER_CHUNKS = 32
MAX_PLAYBACK_BUFFER_AGE = 3  # seconds

class AudioDevice:
    def __init__(self, format=FORMAT, rate=RATE, channels=CHANNELS, chunk_size=CHUNK):
        self.audio = pyaudio.PyAudio()
        self.mic_index = -1
        self.speaker_index = -1
        self._find_devices()

        self.format = format
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        self.mic_buffer = queue.deque(maxlen=MIC_BUFFER_CHUNKS)
        self.mic_buffer_unfiltered = queue.deque(maxlen=MIC_BUFFER_CHUNKS)
        self.mic_lock = threading.Lock()
        self.mic_buffer_ready = threading.Condition(self.mic_lock)
        self.mic_ready = threading.Event()
        self.read_in_progress = False
        self.read_did_overflow = False
        self.active_chunk = None
        self.active_chunk_position = 0

        self.playback_queue = queue.Queue()
        self.playback_buffer = PlaybackBuffer(self.rate)
        self.playback_buffer_lock = threading.Lock()
        self.playback_ready = threading.Event()        
        self.playback_is_playing = False
        self.playback_last_frame_time = 0

        self.running = True
        self._filtered = True
        self.filtered = True
        self.delay = 0
        
        # Discard the first few microphone frames as the microphone subsystem warms up
        self._mic_warmup_frames = 1024 * 3

        self._start_mic_thread()
        self._start_playback_thread()

        # Wait for both threads to be ready
        self.playback_ready.wait()
        self.mic_ready.wait()

    def set_delay(self, delay):
        self.delay = delay

    def set_sample_delay(self, delay):
        self.delay = delay / self.rate

    def set_filter_mode(self, mode):
        self._filtered = bool(mode)

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
                    format=self.format, channels=self.channels, rate=self.rate,
                    input=True, frames_per_buffer=self.chunk_size,
                    input_device_index=self.mic_index)
            
            while self.running:
                mic_frame = AudioData(stream.read(self.chunk_size, exception_on_overflow=False))

                # Discard the first few frames as the microphone subsystem warms up
                if self._mic_warmup_frames > 0:
                    logger.debug(f"Discarding warmup frame {len(mic_frame)} bytes: Mean: {np.mean(np.abs(mic_frame.as_array())):.2f}")
                    self._mic_warmup_frames -= len(mic_frame)
                    continue

                # Notify that the microphone is ready
                if not self.mic_ready.is_set():
                    self.mic_ready.set()

                if self._filtered:
                    filtered_frame = self._filter_frame(mic_frame)
                else:
                    filtered_frame = mic_frame
                    
                with self.mic_lock:
                    self.mic_buffer_unfiltered.append(mic_frame)
                    if len(self.mic_buffer) == self.mic_buffer.maxlen:
                        self.read_did_overflow = True
                        # Discard active chunk as it's stale
                        if self.active_chunk is not None:
                            self.active_chunk = None
                            self.active_chunk_position = 0

                    self.mic_buffer.append(filtered_frame.as_array())
                    self.mic_buffer_ready.notify_all()
            stream.stop_stream()
            stream.close()

        threading.Thread(target=mic_worker, daemon=True).start()

    def _start_playback_thread(self):
        def playback_worker():

            # TODO: allow playback to be at a different rate than the microphone
            with TimeThis("Playback thread init"):
                stream = self.audio.open(
                    format=self.format, channels=self.channels, rate=self.rate,
                    output=True, output_device_index=self.speaker_index)

            while self.running:
                try:
                    # Notify that playback is ready
                    if not self.playback_ready.is_set():
                        self.playback_ready.set()
                    frames = self.playback_queue.get(timeout=0.1)

                    timestamp = time.perf_counter()
                    # Align frame to the end of the last frame if sufficiently consecutive
                    if timestamp - self.playback_last_frame_time < self.chunk_size / self.rate:
                        timestamp = self.playback_last_frame_time

                    frame_data = AudioData(frames, timestamp=timestamp)
                    logger.debug(f"Playback Frame: {frame_data.timestamp:.3f} - {frame_data.end_time():.3f}")
                    self.playback_last_frame_time = frame_data.end_time()
                    with self.playback_buffer_lock:
                        self.playback_buffer.append(frame_data)
                        self.playback_buffer.prune_older_than(time.perf_counter() - MAX_PLAYBACK_BUFFER_AGE)
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
        nperseg = self.chunk_size  # or another size you prefer
        noverlap = int(nperseg * 0.75)  # or another % you prefer

        delay_sec = self.delay

        frame_start = mic_frame.timestamp + delay_sec
        frame_end = frame_start + mic_frame.duration()

        with self.playback_buffer_lock:
            np_playback = self.playback_buffer.extract_frames(frame_start, len(mic_frame))

        # Log mic average volume
        mic_avg = np.mean(np.abs(mic_frame.as_array()))
        logger.debug(f"Mic Frame: {mic_frame.timestamp:.3f} - {mic_frame.end_time():.3f} ({mic_avg:.1f})")

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
        _, _, S_playback = stft(np_playback, fs=self.rate, nperseg=nperseg, noverlap=noverlap)
        _, _, S_mic = stft(np_mic, fs=self.rate, nperseg=nperseg, noverlap=noverlap)

        # Ensure signals are same length in frequency domain
        min_frames = min(S_playback.shape[1], S_mic.shape[1])
        S_playback = S_playback[:, :min_frames]
        S_mic = S_mic[:, :min_frames]

        # Subtracting spectral enegy of playback from microphone
        cleaned_magnitudes = np.maximum(np.abs(S_mic) - np.abs(S_playback), 0)
        cleaned_spectrum = cleaned_magnitudes * np.exp(1j * np.angle(S_mic))

        # Inverse Short-time Fourier Transform - Transform back to time domain
        _, cleaned_time = istft(cleaned_spectrum, fs=self.rate, nperseg=nperseg, noverlap=noverlap)

        if np.any(np.isnan(cleaned_time)):
            logger.warning("NaNs in ISTFT output")
            return mic_frame

        # Clip the cleaned audio to the range of int16 and convert back to original format
        np_type = mic_frame.get_np_type()
        min_range = np.iinfo(np_type).min
        max_range = np.iinfo(np_type).max
        cleaned_clip = np.clip(cleaned_time, min_range, max_range).astype(np_type)

        return AudioData(cleaned_clip, format=mic_frame.format, channels=mic_frame.channels, rate=mic_frame.rate)

    def read(self, size: int = None) -> np.ndarray:
        result = []
        if size is None:
            size = self.chunk_size
        frames_remaining = size

        if size <= 0:
            raise ValueError("Size must be greater than 0")
        
        with self.mic_buffer_ready:
            while frames_remaining > 0:
                # If we have an active chunk, copy from it
                # The existance of an active chunk implies we're reading and have not
                # overflowed the buffer.
                if self.active_chunk is not None:
                    # Copy from the active chunk
                    frames_available = len(self.active_chunk) - self.active_chunk_position
                    frame_count = min(frames_remaining, frames_available)
                    segment = self.active_chunk[self.active_chunk_position : self.active_chunk_position + frame_count]

                    # Is the active chunk exhausted?
                    if self.active_chunk_position + frame_count >= len(self.active_chunk):
                        self.active_chunk = None
                        self.active_chunk_position = 0
                    else:
                        # Update the active chunk position
                        self.active_chunk_position += frame_count

                    result.append(segment)
                    frames_remaining -= frame_count
                else:
                    # We need more chunks
                    while len(self.mic_buffer) == 0:
                        # Wait for a signal that a chunk is ready
                        self.mic_buffer_ready.wait()

                    # Are we reading and current with the buffer?
                    if self.read_in_progress and not self.read_did_overflow:
                        self.active_chunk = self.mic_buffer.popleft()
                    elif not self.read_in_progress:
                        # Start with the most recent chunk
                        self.active_chunk = self.mic_buffer.pop()
                        self.mic_buffer.clear()
                        self.mic_buffer_unfiltered.clear()
                    else:
                        # TODO: communicate buffer overflow to caller
                        logger.warning("Read Buffer overflow detected, discarding frames")
                        # On an overflow discard as many frames as possible to
                        # catch up with the requested frames
                        chunks_needed = ceil(frames_remaining / self.chunk_size)

                        # Grab the rightmost chunks_needed chunks
                        chunks = list(self.mic_buffer)[-chunks_needed:]
                        self.mic_buffer.clear()
                        self.mic_buffer_unfiltered.clear()

                        temp_chunk = np.concatenate(chunks)

                        # Place the last frames_remaining frames in the active chunk
                        self.active_chunk = temp_chunk[-frames_remaining:]
                        self.active_chunk_position = 0

                        # Clear overflow
                        self.read_did_overflow = False

                    self.read_in_progress = True

            return np.concatenate(result)

    def reset_microphone(self):
        with self.mic_lock:
            self.mic_buffer.clear()
            self.mic_buffer_unfiltered.clear()
            self.read_in_progress = False
            self.read_did_overflow = False

    def play(self, audio_data):
        if isinstance(audio_data, AudioData):
            assert audio_data.format == self.format
            audio_data = audio_data.as_bytes()

        sample_size = pyaudio.get_sample_size(self.format)
        bytes_per_chunk = self.chunk_size * sample_size
        total_length = len(audio_data)
        
        chunks_queued = 0
        for i in range(0, total_length, bytes_per_chunk):
            chunk = audio_data[i:i + bytes_per_chunk]
            self.playback_queue.put(chunk)
            chunks_queued += 1

        logger.debug(f"Queued {chunks_queued} chunks for playback")

    def stop(self):
        self.running = False
        time.sleep(0.5)
        self.audio.terminate()
