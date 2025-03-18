import logging
import numpy as np
import pyaudio
from pydub import AudioSegment
import wave

from config import (
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS
)

logger = logging.getLogger(__name__)

class AudioBuffer:
    def __init__(self, audio_format, sample_rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS):
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._channels = channels

        self._frame_size = pyaudio.get_sample_size(audio_format) * self._channels
        self.chunks = []

    def append(self, chunk):
        chunk_size = len(chunk)
        if chunk_size % self._frame_size != 0:
            logger.warning(f"Chunk size {chunk_size} is not a multiple of frame size {self._frame_size}")
            # drop last few samples
            chunk = chunk[:-(chunk_size % self._frame_size)]

        self.chunks.append(chunk)

    def get_bytes(self):
        return b"".join(self.chunks)

    def get_frames(self):
        np_dtype = self._frame_size_to_numpy()
        audio_data = np.frombuffer(self.get_bytes(), dtype=np_dtype)

        if self._channels > 1:
            audio_data = audio_data.reshape(-1, self._channels)
        return audio_data

    def get_duration_ms(self):
        return self.frame_count() / self._sample_rate * 1000

    def byte_count(self):
        return sum(len(chunk) for chunk in self.chunks)

    def frame_count(self):
        return sum(len(chunk) // self._frame_size for chunk in self.chunks)

    def clear(self):
        self.chunks.clear()

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
        
def save_recording(recording, filename):
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    # Create wave file from audio bytes using pyaudio
    with wave.open(str(filename), "wb") as wave_file:
        wave_file.setnchannels(AUDIO_CHANNELS)
        wave_file.setsampwidth(pyaudio.get_sample_size(AUDIO_FORMAT))
        wave_file.setframerate(AUDIO_SAMPLE_RATE)
        wave_file.writeframes(recording)
    
    logger.info(f"Recording saved to {filename}")


def save_recording_mp3(recording, filename):
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)


    if not isinstance(recording, AudioSegment):
        recording = AudioSegment(
            data=recording,
            sample_width=pyaudio.get_sample_size(AUDIO_FORMAT),
            frame_rate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS
        )
    
    with open(filename, "wb") as mp3_file:
        recording.export(mp3_file, format="mp3")


    logger.info(f"Recording saved to {filename}")