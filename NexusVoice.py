import logging
import numpy as np
import openwakeword
from openwakeword.model import Model
import os
import pyaudio

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

AUDIO_FORMAT = pyaudio.paInt16
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
# Models are trained on 0.08s audio chunks
AUDIO_CHUNK = 0.08 * AUDIO_RATE

INFERENCE_FRAMEWORK = "onnx"
ACTIVATION_THRESHOLD = 0.5


def initialize_openwakeword():
    model_paths = openwakeword.get_pretrained_model_paths()

    all_paths_exist = all(os.path.exists(model_path) for model_path in model_paths)
    if not all_paths_exist:
        logger.info("Downloading OpenWakeWord models...")
        openwakeword.utils.download_models()

class NexusVoice:
    def __init__(self):
        self.audio_stream = None

        models = [
            "hey jarvis",
            "./models/my_nexus-3.onnx",
            "./models/my_nexus.onnx",
            #"./models/scarlette.onnx" # model seems broken
        ]
        self.model = Model(wakeword_models=models, enable_speex_noise_suppression=True, inference_framework=INFERENCE_FRAMEWORK)

    def run(self):
        logger.info("NexusVoice is listening...")

        self.open_audio_stream()

        while True:
            raw_audio = self.audio_stream.read(int(AUDIO_CHUNK))
            np_audio = np.frombuffer(raw_audio, dtype=np.int16)

            detection = self.model.predict(np_audio)
            # print prediction values with 2 decimal places on one line repeatedly
            print(" ".join([f"{key}: {value:.2f}" for key, value in detection.items()]), end="\r")
            if any(value > ACTIVATION_THRESHOLD for value in detection.values()):
                detected_wake_words = [key for key, value in detection.items() if value > ACTIVATION_THRESHOLD]
                print(f"\nWake word detected! {detected_wake_words}")
                self.reset_model()

    def open_audio_stream(self):
        self.audio_stream = pyaudio.PyAudio().open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=int(AUDIO_CHUNK)
        )

    def reset_model(self):
        # Pipe silence into model to reset wake word detection
        for _ in range(4):
            silence = np.zeros(int(AUDIO_CHUNK), dtype=np.int16)
            self.model.predict(silence)

def main():
    initialize_openwakeword()

    app = NexusVoice()

    app.run()

if __name__ == '__main__':
    main()