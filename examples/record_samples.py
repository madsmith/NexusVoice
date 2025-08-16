import pyaudio
import silero_vad
import torch

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - '%(message)s'")

from nexusvoice.audio import AudioDevice
from nexusvoice.audio.utils import AudioBuffer, AudioData, save_recording, save_recording_mp3

AUDIO_FORMAT = pyaudio.paInt16
SILERO_VAD_AUDIO_CHUNK = 512
RATE = 16000
CHANNELS = 1

def main():
    # Initialize the AudioDevice
    device = AudioDevice()
    device.set_sample_delay(-2200)
    device.initialize()
    device.start()

    # Initialize the Silero VAD
    vad_model = silero_vad.load_silero_vad()

    # Listen to the microphone
    speech_buffer = AudioBuffer(format=AUDIO_FORMAT)
    is_recording = False
    speech_pause_duration = 0.5
    pause_duration = 0

    while True:
        read_size = SILERO_VAD_AUDIO_CHUNK
        chunk = device.read(read_size)

        # Convert the chunk to a tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.int16).float()
        chunk_tensor /= torch.iinfo(torch.int16).max

        vad_score = vad_model(chunk_tensor, 16000).item()
        is_speech = vad_score > 0.5

        if is_speech:
            is_recording = True
            pause_duration = 0
        else:
            pause_duration += read_size / RATE

        if is_recording:
            speech_buffer.append(chunk)

            if pause_duration >= speech_pause_duration:
                save_recording(speech_buffer, "speech.wav")
                save_recording_mp3(speech_buffer, "speech.mp3")
                print("Recording saved.")
                print("Playing back recording...")
                recording = AudioData(speech_buffer.get_frames(), AUDIO_FORMAT)
                device.play(recording)
                # time.sleep(recording.duration() + 0.5)
                speech_buffer.clear()
                device.reset_microphone()
                is_recording = False

if __name__ == "__main__":
    main()