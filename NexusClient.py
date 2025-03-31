import io
import logging
import queue
import threading
import time

import numpy as np
import omegaconf
from openwakeword.model import Model as OpenWakeWordModel
import pyaudio
import pydub
import silero_vad
import torch


from audio_utils import AudioBuffer, AudioRingBuffer
from config import (
    AUDIO_FORMAT,
    NUMPY_AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_CHUNK,
    VAD_SILENCE_DURATION,
    VAD_ACTIVATION_THRESHOLD,
    ACTIVATION_THRESHOLD
)

WAKE_WORD_FRAME_CHUNK = 1280
SILERO_VAD_AUDIO_CHUNK = 512

INFERENCE_FRAMEWORK = "onnx"

logger = logging.getLogger(__name__)

# Add TRACE log level
LOG_TRACE_LEVEL = logging.DEBUG // 2
logging.addLevelName(LOG_TRACE_LEVEL, "TRACE")
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(LOG_TRACE_LEVEL):
        self._log(LOG_TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace

class NexusVoiceClient(threading.Thread):
    class ListenThread(threading.Thread):
        def __init__(self, client):
            thread_name = f"{client.name}::listen"
            super().__init__(daemon=True, name=thread_name)

            self.client = client
            self.running = False
            self._isRecording = threading.Event()
            self.audio_stream = None
            self.wake_word_buffer = None
            self.vad_buffer = None
            self.speech_buffer = None
            self.audio_is_speech = False
            self.silence_duration = 0

        def run(self):
            logger.info("Starting listen thread")

            self.initialize()

            self.running = True
            try:
                logger.info("Listening for audio")
                self._open_audio_stream()
                
                while self.running:
                    try:
                        # How much audio is needed
                        wake_word_needed = WAKE_WORD_FRAME_CHUNK - self.wake_word_buffer.frame_count()
                        #vad_needed = VAD_AUDIO_CHUNK - vad_buffer.frame_count()
                        vad_needed = SILERO_VAD_AUDIO_CHUNK - self.vad_buffer.frame_count()

                        # Determine how much audio is needed to fill one of the buffers
                        read_chunk_size = min(wake_word_needed, vad_needed)

                        audio_bytes = self.audio_stream.read(read_chunk_size, exception_on_overflow=False)
                        logger.trace(f"Read {len(audio_bytes)} bytes for {read_chunk_size} frames")

                        self.wake_word_buffer.append(audio_bytes)
                        self.vad_buffer.append(audio_bytes)

                        if self.isRecording():
                            self.client.add_command(NexusVoiceClient.CommandProcessAudio(audio_bytes))

                        self.process_wake_word()
                        self.process_vad()

                        if self.audio_is_speech:
                            self.speech_buffer.append(audio_bytes)
                        else:
                            self.speech_buffer.clear()
                    except Exception as e:
                        logger.error(e)

            except Exception as e:
                logger.error(e)
            finally:
                self._close_audio_stream()

        def stop(self):
            logger.info("Stopping listen thread")
            self.running = False

        def _open_audio_stream(self):
            logger.debug("Opening audio stream")
            self.audio_stream = pyaudio.PyAudio().open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=SILERO_VAD_AUDIO_CHUNK)
    
        def _close_audio_stream(self):
            if self.audio_stream:
                logger.debug("Closing audio stream")
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

        def initialize(self):
            self._initializeWakeWordModel()
            self._initializeVADModel()
            self._initializeBuffers()

        def _initializeWakeWordModel(self):
            logger.info("Initializing wake word model")
            models = [
                "hey jarvis",
                "./models/nexus_v4.onnx",
                "models/stop.onnx"
            ]
            self.wake_word_model = OpenWakeWordModel(
                wakeword_models=models,
                inference_framework=INFERENCE_FRAMEWORK,
                enable_speex_noise_suppression=True
            )

        def _initializeVADModel(self):
            logger.info("Initializing VAD model")
            self.vad_model = silero_vad.load_silero_vad()

        def _initializeBuffers(self):
            self.wake_word_buffer = AudioBuffer(AUDIO_FORMAT)
            self.vad_buffer = AudioBuffer(AUDIO_FORMAT)
            self.speech_buffer = AudioRingBuffer(AUDIO_FORMAT, max_duration=10.0)

        def process_wake_word(self):
            while self.wake_word_buffer.frame_count() >= WAKE_WORD_FRAME_CHUNK:
                logger.trace("Processing wake word buffer")
                audio_frames = self.wake_word_buffer.get_frames()
                self.wake_word_buffer.clear()

                # push back any extra frames
                if len(audio_frames) > WAKE_WORD_FRAME_CHUNK:
                    extra_frames = audio_frames[WAKE_WORD_FRAME_CHUNK:]
                    audio_frames = audio_frames[:WAKE_WORD_FRAME_CHUNK]
                    self.wake_word_buffer.append(extra_frames)

                # oww expects numpy array
                np_audio = np.frombuffer(audio_frames, dtype=NUMPY_AUDIO_FORMAT)

                detection = self.wake_word_model.predict(np_audio)

                if any(value > ACTIVATION_THRESHOLD for value in detection.values()):
                    detected_wake_words = [key for key, value in detection.items() if value > ACTIVATION_THRESHOLD]
                    logger.debug(f"\nWake word detected! {detected_wake_words}")
                    self.wake_word_model.reset()
                    for word in detected_wake_words:
                        self.client.add_command(NexusVoiceClient.CommandWakeWord(word, self.speech_buffer.get_bytes()))
 
        def process_vad(self):
            while self.vad_buffer.frame_count() >= SILERO_VAD_AUDIO_CHUNK:
                logger.trace("Processing VAD buffer")
                audio_frames = self.vad_buffer.get_frames()
                self.vad_buffer.clear()

                # push back any extra frames
                if len(audio_frames) > SILERO_VAD_AUDIO_CHUNK:
                    extra_frames = audio_frames[SILERO_VAD_AUDIO_CHUNK:]
                    audio_frames = audio_frames[:SILERO_VAD_AUDIO_CHUNK]
                    self.vad_buffer.append(extra_frames)

                # silero vad expects a tensor (normalized to -1 to 1)
                torch_audio = torch.tensor(list(audio_frames), dtype=torch.int16).float()
                torch_audio /= torch.iinfo(torch.int16).max

                vad_score = self.vad_model(torch_audio, AUDIO_SAMPLE_RATE).item()

                self.audio_is_speech = vad_score > VAD_ACTIVATION_THRESHOLD
                if self.audio_is_speech:
                    logger.trace(f"VAD score: {vad_score} - Speech: {self.audio_is_speech}")
                    self.silence_duration = 0
                else:
                    self.silence_duration += SILERO_VAD_AUDIO_CHUNK / AUDIO_SAMPLE_RATE

                if self.isRecording() and self.silence_duration > VAD_SILENCE_DURATION:
                    logger.debug(f"VAD silence duration {self.silence_duration} exceeded threshold")
                    self.stopRecording()

        def isRecording(self):
            return self._isRecording.is_set()
        
        def startRecording(self):
            self._isRecording.set()

        def stopRecording(self):
            self._isRecording.clear()

    class PlaybackThread(threading.Thread):
        def __init__(self, client):
            thread_name = f"{client.name}::playback"
            super().__init__(daemon=True, name=thread_name)

            self.client = client
            self.running = False
            self.command_queue = queue.Queue()

        def run(self):
            logger.info("Starting playback thread")
            self.running = True
            try:
                while self.running:
                    try:
                        try:
                            command = self.command_queue.get(timeout=1)
                        except queue.Empty:
                            continue
                        logger.debug(f"Received command {command}")
                        
                        if isinstance(command, NexusVoiceClient.CommandShutdown):
                            logger.trace("Received shutdown command")
                            break
                        elif isinstance(command, NexusVoiceClient.CommandPlayAudioData):
                            audio_bytes = command.audio_bytes
                            logger.debug(f"Playing audio data {len(audio_bytes)}")
                            self.play_audio_data(audio_bytes, format=command.format)
                        else:
                            logger.warning(f"Unknown command {command}")
                    except queue.Empty:
                        continue
            except Exception as e:
                logger.error(f"Playback error {e} - {type(e)}")
                # print stack trace
                import traceback
                traceback.print_exc()

        def stop(self):
            logger.info("Shutting down playback thread")
            self.running = False
            self.add_command(NexusVoiceClient.CommandShutdown())

        def add_command(self, command):
            self.command_queue.put(command)

        def play_audio_data(self, audio_bytes, format):
            p = pyaudio.PyAudio()
            logger.debug("Opening output audio stream")
            try:
                stream = p.open(format=AUDIO_FORMAT,
                                channels=AUDIO_CHANNELS,
                                rate=AUDIO_SAMPLE_RATE,
                                output=True)
            except Exception as e:
                logger.error(f"Error opening audio stream: {e}")
                return
            
            try:
                # Convert bytearray to bytes-like
                audio_data = io.BytesIO(audio_bytes).read()
                stream.write(audio_data)
            except Exception as e:
                logger.error(f"Error writing audio data: {e}")
            finally:
                stream.stop_stream()
                stream.close()

                p.terminate()

    class Command:
        def __init__(self):
            pass

        def __str__(self):
            return self.__class__.__name__
        
        def __repr__(self):
            return self.__str__()

    class CommandWakeWord(Command):
        def __init__(self, wake_word, speech_bytes = None):
            self.wake_word = wake_word
            self.speech_bytes = speech_bytes

    class CommandShutdown(Command):
        pass

    class CommandProcessAudio(Command):
        def __init__(self, audio_bytes):
            self.audio_bytes = audio_bytes

    class CommandPlayAudioData(Command):
        def __init__(self, audio_bytes, format="pcm"):
            self.audio_bytes = audio_bytes
            self.format = format

    def __init__(self, client_id, config):
        thread_name = f"NexusVoiceClient::{client_id}"
        super().__init__(daemon=True, name=thread_name)

        self.client_id = client_id
        self.config = config
        self.audio_stream = None

        self.playback_thread = None
        self.listen_thread = None

        self.command_queue = queue.Queue()

        self.running = False

    def run(self):
        logger.info(f"Starting NexusVoiceClient {self.client_id}")
        
        # Create microphone and playback threads
        self.playback_thread = NexusVoiceClient.PlaybackThread(self)
        self.listen_thread = NexusVoiceClient.ListenThread(self)

        self.playback_thread.start()
        self.listen_thread.start()

        self.running = True
        try:
            while self.running:
                try:
                    command = self.command_queue.get(timeout=1)
                    print(command)
                    if isinstance(command, NexusVoiceClient.CommandShutdown):
                        # Shutdown command allows the client to recognize that it should stop
                        logger.trace("Received shutdown command")
                        break
                    elif isinstance(command, NexusVoiceClient.CommandWakeWord):
                        logger.debug(f"Received wake word command {command.wake_word}")
                        if command.wake_word == "stop":
                            self.stop()
                        elif command.wake_word == "nexus_v4":
                            # TODO: rename nexus_v4 to nexus
                            self.listen_thread.startRecording()
                        else:
                            logger.warning(f"Unknown wake word {command.wake_word}")

                        if command.speech_bytes:
                            self.playback_thread.add_command(NexusVoiceClient.CommandPlayAudioData(command.speech_bytes, "pcm"))

                    elif isinstance(command, NexusVoiceClient.CommandProcessAudio):
                        audio_bytes = command.audio_bytes
                        logger.trace(f"Received audio bytes {len(audio_bytes)}")
                    else:
                        logger.warning(f"Unknown command {command}")

                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error in NexusVoiceClient: {e}")
            

    def stop(self):
        logger.info(f"Stopping NexusVoiceClient {self.client_id}")
        self.running = False
        self.add_command(NexusVoiceClient.CommandShutdown())
        if self.playback_thread:
            self.playback_thread.stop()
            self.playback_thread.join()
        if self.listen_thread:
            self.listen_thread.stop()
            self.listen_thread.join()

    def add_command(self, command):
        self.command_queue.put(command)

def main():
    try:

        # logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        log_format = "[%(asctime)s] [%(levelname)s] - %(name)s %(message)s"
        log_format = "[{levelname}]\t{threadName}\t{message}"
        log_level = logging.DEBUG
        #log_level = LOG_TRACE_LEVEL
        logging.basicConfig(level=log_level, style="{", format=log_format)

        logger.debug("Loading config")
        config = omegaconf.OmegaConf.load("config.yml")

        logger.debug("Creating NexusVoiceClient")
        client = NexusVoiceClient("test", config)
        client.start()

        client.join()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    main()