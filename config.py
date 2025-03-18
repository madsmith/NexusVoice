import pyaudio
import numpy as np

AUDIO_FORMAT = pyaudio.paInt16
NUMPY_AUDIO_FORMAT = np.int16
# OpenWakeWord models are trained on 16kHz audio
# VAD supports 8kHz, 16kHz, 32kHz, 48kHz
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
# Models are trained on 0.08s audio chunks
WAKE_WORD_AUDIO_CHUNK = 1280 # 0.08 * AUDIO_RATE
# VAD supports 10ms, 20ms and 30ms audio chunks
VAD_AUDIO_CHUNK = 480 # 0.03 * AUDIO_RATE
SILERO_VAD_AUDIO_CHUNK = 512
# pyaudio buffer size should be GCD of WAKE_WORD_AUDIO_CHUNK and VAD_AUDIO_CHUNK
AUDIO_CHUNK = 160 # 0.01 * AUDIO_RATE

VAD_SILENCE_DURATION = 1.5
VAD_ACTIVATION_THRESHOLD = 0.5

INFERENCE_FRAMEWORK = "onnx"
ACTIVATION_THRESHOLD = 0.5