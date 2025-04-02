from abc import ABC, abstractmethod
from collections import deque
import logging
from pathlib import Path
import numpy as np
import pyaudio
from pydub import AudioSegment
import wave

from config import (
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS
)

from utils.bytes import ByteRingBuffer

logger = logging.getLogger(__name__)


class AudioBufferBase(ABC):
    def __init__(self, audio_format, sample_rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS):
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_size = pyaudio.get_sample_size(audio_format) * self._channels

    def get_channels(self):
        return self._channels
    
    def get_sample_rate(self):
        return self._sample_rate
    
    def get_audio_format(self):
        return self._audio_format
    
    def get_sample_size(self):
        return pyaudio.get_sample_size(self._audio_format)

    @abstractmethod
    def append(self, chunk):
        pass

    @abstractmethod
    def get_bytes(self):
        pass

    @abstractmethod
    def byte_count(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    def __len__(self):
        return self.frame_count()

    def frame_count(self):
        return self.byte_count() // self._frame_size

    def get_frames(self):
        np_dtype = self._frame_size_to_numpy()
        audio_data = np.frombuffer(self.get_bytes(), dtype=np_dtype)

        if self._channels > 1:
            audio_data = audio_data.reshape(-1, self._channels)
        return audio_data

    def get_duration_ms(self):
        return self.frame_count() / self._sample_rate * 1000

    def _frame_size_to_numpy(self):
        # convert frame size to numpy dtype
        if self._frame_size == 1:
            return np.uint8
        elif self._frame_size == 2:
            return np.int16
        elif self._frame_size == 4:
            return np.int32
        else:
            raise ValueError(f"Unsupported frame size {self._frame_size}")

class AudioBuffer(AudioBufferBase):
    def __init__(self, audio_format, sample_rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS):
        super().__init__(audio_format, sample_rate, channels)
        self.chunks = []

    def append(self, chunk):
        if isinstance(chunk, np.ndarray):
            chunk = chunk.tobytes()

        chunk_size = len(chunk)
        if chunk_size % self._frame_size != 0:
            logger.warning(f"Chunk size {chunk_size} is not a multiple of frame size {self._frame_size}")
            # drop last few samples
            chunk = chunk[:-(chunk_size % self._frame_size)]

        self.chunks.append(chunk)

    def get_bytes(self):
        return b"".join(self.chunks)

    def byte_count(self):
        return sum(len(chunk) for chunk in self.chunks)

    def clear(self):
        self.chunks.clear()
        # convert frame size to numpy dtype
        if self._frame_size == 1:
            return np.uint8
        elif self._frame_size == 2:
            return np.int16
        elif self._frame_size == 4:
            return np.int32
        else:
            raise ValueError(f"Unsupported frame size {self._frame_size}")

class AudioRingBuffer(AudioBufferBase):
    def __init__(self, audio_format, sample_rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, max_duration=1.0):
        super().__init__(audio_format, sample_rate, channels)
        self.max_duration = max_duration
        buffer_size = int(max_duration * sample_rate * self._frame_size)
        self.buffer = ByteRingBuffer(buffer_size)

    def append(self, chunk):
        if isinstance(chunk, np.ndarray):
            chunk = chunk.tobytes()
        self.buffer.append(chunk)

    def get_bytes(self):
        return self.buffer.get_bytes()
    
    def byte_count(self):
        return self.buffer.byte_count()
    
    def clear(self):
        self.buffer.clear()

def save_recording(recording, filename):
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(recording, AudioBuffer):
        frames = recording.get_bytes()
        sample_rate = recording.get_sample_rate()
        format = recording.get_audio_format()
        channels = recording.get_channels()
    else:
        frames = recording
        sample_rate = AUDIO_SAMPLE_RATE
        format = AUDIO_FORMAT
        channels = AUDIO_CHANNELS

    sample_width = pyaudio.get_sample_size(format)

    # Create wave file from audio bytes using pyaudio
    with wave.open(str(filename), "wb") as wave_file:
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(frames)
    
    logger.info(f"Recording saved to {filename}")

def save_recording_mp3(recording, filename):
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(recording, AudioBuffer):
        frames = recording.get_bytes()
        sample_rate = recording.get_sample_rate()
        format = recording.get_audio_format()
        channels = recording.get_channels()
        recording = AudioSegment(
            data=frames,
            sample_width=pyaudio.get_sample_size(format),
            frame_rate=sample_rate,
            channels=channels
        )
    elif not isinstance(recording, AudioSegment):
        recording = AudioSegment(
            data=recording,
            sample_width=pyaudio.get_sample_size(AUDIO_FORMAT),
            frame_rate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS
        )
    
    with open(filename, "wb") as mp3_file:
        recording.export(mp3_file, format="mp3")


    logger.info(f"Recording saved to {filename}")



# Below is a seperate implementation the resembles some of the above features
# I should consolidate these into a more coherent approach at some point.

import numpy as np
import time
import pyaudio

class AudioData:
    def __init__(self, frames: bytes, format=pyaudio.paInt16, channels=1, rate=16000, timestamp=None):
        assert isinstance(frames, bytes) or isinstance(frames, np.ndarray), "frames must be of type bytes or numpy ndarray"
        # Format describes how many audio bytes are in each sample
        self.format = format
        self.channels = channels
        self.rate = rate
        if isinstance(frames, np.ndarray):
            self.frames = frames.tobytes()
        else:
            self.frames = frames
        self.timestamp = timestamp
        if not timestamp:
            self.timestamp = time.perf_counter()

    def as_bytes(self) -> bytes:
        """ Return raw audio bytes """
        return self.frames
    
    def as_array(self, dtype=None) -> np.ndarray:
        """ Convert bytes to numpy array """
        sample_dtype = self.get_np_type()

        np_array = np.frombuffer(self.frames, dtype=sample_dtype)

        if self.channels > 1:
            np_array = np_array.reshape(-1, self.channels)

        if dtype is not None and dtype != sample_dtype:
            np_array = np_array.astype(dtype)

        return np_array
    
    def __len__(self):
        return self.frame_count()

    def frame_count(self) -> int:
        return len(self.frames) // pyaudio.get_sample_size(self.format) // self.channels
    
    def duration(self) -> float:
        return self.frame_count() / self.rate

    def end_time(self) -> float:
        return self.timestamp + self.duration()
    
    def get_np_type(self):
        sample_size = pyaudio.get_sample_size(self.format)
        if sample_size == 1:
            return np.uint8
        elif sample_size == 2:
            return np.int16
        elif sample_size == 4:
            return np.int32
        elif sample_size == 8:
            return np.float64
        else:
            raise ValueError(f"Unsupported sample size {sample_size}")

    @classmethod
    def sample_width_to_format(cls, sample_width: int):
        if sample_width == 1:
            return pyaudio.paInt8
        elif sample_width == 2:
            return pyaudio.paInt16
        elif sample_width == 4:
            return pyaudio.paInt32
        elif sample_width == 8:
            return pyaudio.paFloat64
        else:
            raise ValueError(f"Unsupported sample width {sample_width}")

    @classmethod
    def from_wave(cls, filename):
        if isinstance(filename, Path):
            filename = str(filename)

        audio = pyaudio.PyAudio()
        with wave.open(filename, 'rb') as wf:
            sample_width = wf.getsampwidth()
            frame_bytes = wf.readframes(wf.getnframes())
            audio_data = cls(
                frame_bytes,
                format=cls.sample_width_to_format(sample_width),
                channels=wf.getnchannels(),
                rate=wf.getframerate()
            )
        audio.terminate()

        return audio_data

class PlaybackBuffer:
    def __init__(self, rate: int):
        self.rate = rate
        self.chunks: list[AudioData] = []

    def append(self, frames: AudioData):
        assert isinstance(frames, AudioData), "frames must be of type AudioData"
        self.chunks.append(frames)

    def prune_older_than(self, someTime: float):
        self.chunks = [chunk for chunk in self.chunks if chunk.end_time() > someTime]

    def __len__(self):
        return len(self.chunks)
    
    def extract_frames(self, start: float, frame_count: int) -> np.ndarray:
        """
        Extracts `frame_count` samples of audio starting from `start` time.
        Pads with zeros where data is missing.
        """
        output = np.zeros(frame_count, dtype=np.int16)
        end = start + (frame_count / self.rate)

        for chunk in self.chunks:
            chunk_start = chunk.timestamp
            chunk_end = chunk.end_time()

            if chunk_end < start:
                continue
            if chunk_start > end:
                continue

            chunk_data = chunk.as_array()

            # Calculate overlap
            overlap_start_time = max(start, chunk_start)
            overlap_end_time = min(end, chunk_end)

            out_start_idx = int((overlap_start_time - start) * self.rate)
            out_end_idx = int((overlap_end_time - start) * self.rate)

            in_start_idx = int((overlap_start_time - chunk_start) * self.rate)
            in_end_idx = int((overlap_end_time - chunk_start) * self.rate)

            copy_len = min(out_end_idx - out_start_idx, in_end_idx - in_start_idx)

            output[out_start_idx:out_start_idx + copy_len] = chunk_data[in_start_idx:in_start_idx + copy_len]

        return output

    def extract_window(self, start: float, end: float) -> np.ndarray:
        """
        Extracts a continuous buffer of audio corresponding to the playback window [start, end].
        If there are gaps in the chunks, fills with zeros.
        """
        total_samples = int((end - start) * self.rate)
        output = np.zeros(total_samples, dtype=np.int16)

        for chunk in self.chunks:
            chunk_start = chunk.timestamp
            chunk_end = chunk.end_time()

            if chunk_end < start:
                continue
            if chunk_start > end:
                continue

            chunk_data = chunk.as_array()

            # Calculate overlap
            overlap_start_time = max(start, chunk_start)
            overlap_end_time = min(end, chunk_end)

            out_start_idx = int((overlap_start_time - start) * self.rate)
            out_end_idx = int((overlap_end_time - start) * self.rate)

            in_start_idx = int((overlap_start_time - chunk_start) * self.rate)
            in_end_idx = int((overlap_end_time - chunk_start) * self.rate)

            # Calculate actual number of samples to copy (ensure bounds agree)
            copy_len = min(out_end_idx - out_start_idx, in_end_idx - in_start_idx)

            output[out_start_idx:out_start_idx + copy_len] = chunk_data[in_start_idx:in_start_idx + copy_len]

        return output

    def dump_windows(self, prefix=""):
        """
        Print the start and end times of each series of contiguous chunks of audio data
        """
        output = ""

        if len(self.chunks) == 0:
            return f"{prefix}No audio data"

        start_time = self.chunks[0].timestamp
        end_time = self.chunks[0].end_time()
        chunk_count = 1

        for chunk in self.chunks[1:]:
            # Check for a gap 
            gap = chunk.timestamp - end_time
            if gap > 1/self.rate:
                output += f"{prefix}Playback Window: {start_time:.3f} - {end_time:.3f} [{chunk_count} chunks]\n"
                start_time = chunk.timestamp
                chunk_count = 0
            chunk_count += 1
            end_time = max(end_time, chunk.end_time())

        output += f"{prefix}Playback Window: {start_time:.3f} - {end_time:.3f} [{chunk_count} chunks]"
        
        return output