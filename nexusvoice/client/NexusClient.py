import nexusvoice.bootstrap

import queue
import threading
import numpy as np
import omegaconf
from openwakeword.model import Model as OpenWakeWordModel
import pyaudio
import silero_vad
import torch
import torchaudio

from nexusvoice.ai.AudioInferenceEngine import AudioInferenceEngine
from nexusvoice.ai.TTSInferenceEngine import TTSInferenceEngine
from nexusvoice.audio.utils import AudioBuffer
from nexusvoice.audio.AudioDevice import AudioDevice
from nexusvoice.core.api import NexusAPI, NexusAPILocal

AUDIO_FORMAT = pyaudio.paInt16
NUMPY_AUDIO_FORMAT = np.int16
# OpenWakeWord models are trained on 16kHz audio
# VAD supports 8kHz, 16kHz, 32kHz, 48kHz
AUDIO_SAMPLE_RATE = 16000
# Models are trained on 0.08s audio chunks
WAKE_WORD_FRAME_CHUNK = 1280 # 0.08 * AUDIO_RATE
# VAD supports 10ms, 20ms and 30ms audio chunks
WEBRTC_VAD_AUDIO_CHUNK = 480 # 0.03 * AUDIO_RATE
SILERO_VAD_AUDIO_CHUNK = 512
AUDIO_CHUNK = 160 # 0.01 * AUDIO_RATE

VAD_SILENCE_DURATION = 1.5
VAD_ACTIVATION_THRESHOLD = 0.5

INFERENCE_FRAMEWORK = "onnx"
ACTIVATION_THRESHOLD = 0.5


from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

class NexusVoiceClient(threading.Thread):
    def __init__(self, client_id: str, config: omegaconf.DictConfig):
        thread_name = self._get_thread_name(client_id)
        super().__init__(daemon=True, name=thread_name)

        self.client_id = client_id
        self.config = config

        self._api: NexusAPI = None

        self.audio_device: AudioDevice = None
        self.command_processor: threading.Thread = None

        self.wake_word_model: OpenWakeWordModel = None
        self._vad_model = None
        self._whisper_engine: AudioInferenceEngine = None   
        self._tts_engine: TTSInferenceEngine = None

        self._command_queue = queue.Queue()

        self._vad_buffer: AudioBuffer = None
        self._wake_word_buffer: AudioBuffer = None
        self._speech_buffer: AudioBuffer = None
        
        self._silence_duration = 0
        self._isRecording = threading.Event()

        self.running = False

    def _get_thread_name(self, client_id: str = None):
        return f"NexusVoiceClient::{client_id}"
    
    def get_api(self):
        if self._api is None:
            logger.error("API not initialized")
            raise RuntimeError("API not initialized")
        return self._api
    
    def initialize(self):
        self._initialize_buffers()
        self._initialize_wake_word_model()
        self._initialize_VAD_model()
        self._initialize_whisper_model()
        self._initialize_TTS_model()

        self._initialize_threads()

        self._initialize_audio_device()

    def _initialize_audio_device(self):
        logger.info("Initializing audio device")
        self.audio_device = AudioDevice()
        self.audio_device.set_sample_delay(self.config.audio.get("sample_delay", -2200))

    def _initialize_buffers(self):
        logger.info("Initializing buffers")
        self._wake_word_buffer = AudioBuffer(format=AUDIO_FORMAT)
        self._vad_buffer = AudioBuffer(format=AUDIO_FORMAT)
        self._speech_buffer = AudioBuffer(format=AUDIO_FORMAT)

    def _initialize_wake_word_model(self):
        logger.info("Initializing wake word model")

        model_conf = omegaconf.OmegaConf.select(self.config, "wake_word.models")
        if model_conf is None:
            logger.error("No wake word models found in config")
            raise RuntimeError("No wake word models found in config")

        models = [m.path if "path" in m else m.name for m in model_conf]
        
        self.wake_word_model = OpenWakeWordModel(
            wakeword_models=models,
            inference_framework=INFERENCE_FRAMEWORK,
            enable_speex_noise_suppression=True
        )

    def _initialize_VAD_model(self):
        logger.info("Initializing VAD model")
        self.vad_model = silero_vad.load_silero_vad()

    def _initialize_whisper_model(self):
        logger.info("Initializing Whisper STT...")
        model = self.config.whisper.processor.get("model", "openai/whisper-large-v2")
        self._whisper_engine = AudioInferenceEngine(model)
        self._whisper_engine.initialize()

    def _initialize_TTS_model(self):
        logger.info("Initializing TTS model...")
        voice = self.config.tts.get("voice", None)
        self._tts_engine = TTSInferenceEngine(voices=voice)
        self._tts_engine.initialize()

    def _initialize_threads(self):
        logger.info("Initializing threads")
        thread_name = f"{self.name}::CommandProcessor"
        self.command_processor = threading.Thread(
            name=thread_name,
            target=self.process_commands,
            daemon=True)
        
    def run(self):
        logger.info(f"Starting {self.__class__.__name__} {self.client_id}")

        self.initialize()

        self.running = True

        self.command_processor.start()

        logger.info("Listening for audio")
        while self.running:
            try:
                wake_word_needed = WAKE_WORD_FRAME_CHUNK - self._wake_word_buffer.frame_count()
                vad_needed = SILERO_VAD_AUDIO_CHUNK - self._vad_buffer.frame_count()
                
                # Determine how much audio is needed to fill one of the buffers
                read_chunk_size = min(wake_word_needed, vad_needed)

                audio_bytes = self.audio_device.read(read_chunk_size)

                self._wake_word_buffer.append(audio_bytes)
                self._vad_buffer.append(audio_bytes)

                self._process_vad()
                self._process_wake_word()

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in {self.getName()}: {e}")

        self.stop()

    def process_commands(self):
        """ Process commands from the command queue """
        while self.running:
            try:
                command = self._command_queue.get(timeout=1)
                logger.debug(f"Processing command {command}")
                if isinstance(command, NexusVoiceClient.CommandShutdown):
                    logger.info("Shutting down {self.getName()}")
                    self.stop()
                    break
                elif isinstance(command, NexusVoiceClient.CommandWakeWord):
                    self._process_command_wake_word(command)
                elif isinstance(command, NexusVoiceClient.CommandProcessAudio):
                    self._process_command_process_audio(command)
            except queue.Empty:
                # No command to process
                pass
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                # Show the traceback
                import traceback
                logger.error(traceback.format_exc())

    def _process_command_wake_word(self, command):
        logger.debug(f"Received wake word command {command.wake_word}")
        if not self._confirm_wake_word(command):
            logger.debug(f"Wake word {command.wake_word} not confirmed")
            self.stopRecording(cancel=True)

    def _resample_audio(self, audio_tensor: torch.Tensor, orig_freq=24000, new_freq=16000):
        # Add batch dimension if needed
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(audio_tensor).squeeze(0)  # remove batch dim

    def _tensor_to_int16(self, audio_tensor: torch.Tensor) -> bytes:
        # Clamp values to avoid overflows
        audio_clamped = torch.clamp(audio_tensor, -1.0, 1.0)
        # Convert to int16 range

        max_int16 = np.iinfo(np.int16).max
        audio_int16 = (audio_clamped * max_int16).to(torch.int16)
        return audio_int16.numpy().tobytes()
        
    def _process_command_process_audio(self, command):

        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(command.audio_bytes, dtype=NUMPY_AUDIO_FORMAT)

        transcription = self._whisper_engine.infer(np_audio, AUDIO_SAMPLE_RATE)

        logger.info(f"Transcription: {transcription}")

        response = self.get_api().agent_inference(f"{self.client_id}::agent", transcription)

        logger.info(f"Response: {response} [{type(response)}]")

        audio_tensor = self._tts_engine.infer(response, voice=self.config.tts.voice)

        # Convert the audio tensor to numpy array
        audio = self._tensor_to_int16(self._resample_audio(audio_tensor))

        self.audio_device.play(audio)

    def _process_vad(self):
        """ Process data in the vad buffer if enough data is available"""
        while self._vad_buffer.frame_count() >= SILERO_VAD_AUDIO_CHUNK:
            logger.trace("Processing VAD buffer")
            audio_frames = self._vad_buffer.get_frames()
            self._vad_buffer.clear()

            # push back any extra frames
            if len(audio_frames) > SILERO_VAD_AUDIO_CHUNK:
                extra_frames = audio_frames[SILERO_VAD_AUDIO_CHUNK:]
                audio_frames = audio_frames[:SILERO_VAD_AUDIO_CHUNK]
                self._vad_buffer.append(extra_frames)

            # silero vad expects a tensor (normalized to -1 to 1)
            torch_audio = torch.tensor(list(audio_frames), dtype=torch.int16).float()
            torch_audio /= torch.iinfo(torch.int16).max

            vad_score = self.vad_model(torch_audio, AUDIO_SAMPLE_RATE).item()

            is_speech = vad_score > VAD_ACTIVATION_THRESHOLD
            if is_speech:
                logger.trace(f"VAD score: {vad_score}")
                self._silence_duration = 0
            else:
                self._silence_duration += SILERO_VAD_AUDIO_CHUNK / AUDIO_SAMPLE_RATE

            if is_speech or self._silence_duration < VAD_SILENCE_DURATION or self.isRecording():
                self._speech_buffer.append(audio_frames)
            else:
                self._speech_buffer.clear()

            if self.isRecording() and self._silence_duration > VAD_SILENCE_DURATION:
                logger.debug(f"VAD silence duration {self._silence_duration} exceeded threshold")
                self.stopRecording()

    def _process_wake_word(self):
        if self._wake_word_buffer.frame_count() >= WAKE_WORD_FRAME_CHUNK:
            logger.trace("Processing wake word buffer")
            audio_frames = self._wake_word_buffer.get_frames()
            self._wake_word_buffer.clear()

            # push back any extra frames
            if len(audio_frames) > WAKE_WORD_FRAME_CHUNK:
                extra_frames = audio_frames[WAKE_WORD_FRAME_CHUNK:]
                audio_frames = audio_frames[:WAKE_WORD_FRAME_CHUNK]
                self._wake_word_buffer.append(extra_frames)

            # oww expects numpy array
            np_audio = np.frombuffer(audio_frames, dtype=NUMPY_AUDIO_FORMAT)

            detection = self.wake_word_model.predict(np_audio)

            if any(value > ACTIVATION_THRESHOLD for value in detection.values()):
                detected_wake_words = [key for key, value in detection.items() if value > ACTIVATION_THRESHOLD]
                logger.debug(f"\nWake word detected! {detected_wake_words}")
                self.wake_word_model.reset()
                for word in detected_wake_words:
                    print(f"Detected wake word: {word} VAD: {self._speech_buffer.get_duration_ms()}ms")
                    self.add_command(NexusVoiceClient.CommandWakeWord(word, self._speech_buffer.get_bytes()))
                    self.startRecording()

    def _confirm_wake_word(self, command):
        assert isinstance(command, NexusVoiceClient.CommandWakeWord), "Command is not a wake word command"

        # TODO: Use Misaki to convert the wake word into phonemes and fuzzy match phonemes?

        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(command.audio_bytes, dtype=NUMPY_AUDIO_FORMAT)

        transcription = self._whisper_engine.infer(np_audio, AUDIO_SAMPLE_RATE)

        logger.debug(f"Transcription: {transcription}")

        model_conf = omegaconf.OmegaConf.select(self.config, "wake_word.models")
        if model_conf is None:
            logger.error("No wake word models found in config")
            raise RuntimeError("No wake word models found in config")
        
        # Select the model conf matching the wake word name
        conf = next((m for m in model_conf if m.name == command.wake_word), None)
        if conf is None:
            logger.error(f"No model found for wake word {command.wake_word}")
            return False

        # Remove any special characters
        normalize_transcription = "".join(c.lower() for c in transcription if c.isalnum() or c.isspace())
        
        for phrase in conf.get("valid_phrases", []):
            # Remove any special characters
            normalize_phrase = "".join(c.lower() for c in phrase if c.isalnum() or c.isspace())
            if normalize_phrase in normalize_transcription:
                logger.debug(f"Confirmed wake word {command.wake_word} with phrase {phrase}")
                return True
            
        logger.debug(f"Wake word {command.wake_word} not confirmed with transcription {transcription}")
        return False

    def isRecording(self):
        return self._isRecording.is_set()
    
    def startRecording(self):
        self._isRecording.set()
        logger.debug("Recording started")

    def stopRecording(self, cancel=False):
        self._isRecording.clear()
        if cancel:
            logger.debug("Recording cancelled")
        else:
            logger.debug("Recording stopped")
            audio_data = self._speech_buffer.get_bytes()
            if not audio_data:
                logger.warning("No audio data to process")
            else:
                self.add_command(NexusVoiceClient.CommandProcessAudio(audio_data))

    def stop(self):
        if self.running:
            logger.info(f"Stopping {self.getName()}")
            self.running = False
            
            if self.isRecording():
                self._speech_buffer.clear()
                self.stopRecording()

            self.audio_device.shutdown()

    def add_command(self, command):
        logger.debug(f"Adding command {command}")
        self._command_queue.put(command)

    class Command:
        def __init__(self):
            pass

        def __str__(self):
            return self.__class__.__name__
        
        def __repr__(self):
            return self.__str__()
    
    class CommandShutdown(Command):
        pass

    class CommandWakeWord(Command):
        def __init__(self, wake_word, audio_bytes = None):
            self.wake_word = wake_word
            self.audio_bytes = audio_bytes

    class CommandProcessAudio(Command):
        def __init__(self, audio_bytes):
            self.audio_bytes = audio_bytes

class NexusVoiceStandalone(NexusVoiceClient):
    def __init__(self, client_id: str, config: omegaconf.DictConfig):
        super().__init__(client_id, config)
        self._api = NexusAPILocal(config)

    def _get_thread_name(self, client_id: str = None):
        return f"NexusVoiceStandalone::{client_id}"
    
    def run(self):
        super().run()