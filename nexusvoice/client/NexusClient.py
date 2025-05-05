from typing import Optional

import queue
import threading
import logging
from nexusvoice.core.config import NexusConfig
import numpy as np
from openwakeword.model import Model as OpenWakeWordModel
from pathlib import Path
import pyaudio
import silero_vad
import time
from silero_vad.model import OnnxWrapper
import torch
import torchaudio

from nexusvoice.ai.AudioInferenceEngine import AudioInferenceEngine
from nexusvoice.ai.TTSInferenceEngine import TTSInferenceEngine
from nexusvoice.audio.utils import AudioBuffer, save_recording
from nexusvoice.audio.AudioDevice import AudioDevice
from nexusvoice.client.RecordingState import RecordingState
from nexusvoice.core.api import NexusAPI
from nexusvoice.core.api.online import NexusAPIOnline


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
    def __init__(self, client_id: str, config: NexusConfig):
        thread_name = self._get_thread_name(client_id)
        super().__init__(daemon=True, name=thread_name)

        self.client_id = client_id
        self.config = config

        self._api: Optional[NexusAPI] = None

        self._audio_device: Optional[AudioDevice] = None
        self.command_processor: Optional[threading.Thread] = None

        self._wake_word_model: Optional[OpenWakeWordModel] = None
        self._vad_model: Optional[OnnxWrapper] = None
        self._whisper_engine: Optional[AudioInferenceEngine] = None   
        self._tts_engine: Optional[TTSInferenceEngine] = None
        
        self._command_queue = queue.Queue()

        self._wake_word_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        self._vad_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        self._speech_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        
        self._silence_duration = 0
        self._recording_state = RecordingState()

        self.running = False

    @property
    def audio_device(self) -> AudioDevice:
        assert self._audio_device is not None, f"{self.__class__.__name__} not initialized, audio device not available"
        return self._audio_device

    @property
    def wake_word_model(self) -> OpenWakeWordModel:
        assert self._wake_word_model is not None, f"{self.__class__.__name__} not initialized, wake word model not available"
        return self._wake_word_model

    @property
    def vad_model(self) -> OnnxWrapper:
        assert self._vad_model is not None, f"{self.__class__.__name__} not initialized, vad model not available"
        return self._vad_model

    @property
    def whisper_engine(self) -> AudioInferenceEngine:
        assert self._whisper_engine is not None, f"{self.__class__.__name__} not initialized, whisper engine not available"
        return self._whisper_engine

    @property
    def tts_engine(self) -> TTSInferenceEngine:
        assert self._tts_engine is not None, f"{self.__class__.__name__} not initialized, tts engine not available"
        return self._tts_engine

    @property
    def api(self) -> NexusAPI:
        assert self._api is not None, f"{self.__class__.__name__} not initialized, api not available"
        return self._api
    
    def _get_thread_name(self, client_id: str = "") -> str:
        return f"NexusVoiceClient::{client_id}"

    def initialize(self):
        self._initialize_api()
        self._initialize_wake_word_model()
        self._initialize_VAD_model()
        self._initialize_whisper_model()
        self._initialize_TTS_model()

        self._initialize_threads()

        self._initialize_audio_device()

    def _initialize_api(self):
        logger.info("Initializing API")
        self._api = NexusAPIOnline(self.config)
        self._api.initialize()

    def _initialize_audio_device(self):
        logger.info("Initializing audio device")

        self._audio_device = AudioDevice()
        self._audio_device.set_sample_delay(self.config.get("audio.sample_delay", -2200))

    def _initialize_wake_word_model(self):
        logger.info("Initializing wake word model")

        model_conf = self.config.get("wake_word.models")
        if model_conf is None:
            logger.error("No wake word models found in config")
            raise RuntimeError("No wake word models found in config")

        models = [m.path if "path" in m else m.name for m in model_conf]
        
        self._wake_word_model = OpenWakeWordModel(
            wakeword_models=models,
            inference_framework=INFERENCE_FRAMEWORK,
            enable_speex_noise_suppression=True
        )

    def _initialize_VAD_model(self):
        logger.info("Initializing VAD model")

        vad_model = silero_vad.load_silero_vad()
        if vad_model is None:
            logger.error("Failed to load VAD model")
            raise RuntimeError("Failed to load VAD model")
        self._vad_model = vad_model

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

        assert self.command_processor is not None, "Command processor not initialized"
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
                if logger.isEnabledFor(logging.DEBUG):
                    # Show the traceback
                    import traceback
                    logger.error(traceback.format_exc())

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
                elif isinstance(command, NexusVoiceClient.CommandProcessText):
                    self._process_command_process_text(command)
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
        else:
            logger.trace("Wake word {command.wake_word} confirmed")
            self.startRecording(confirmed=True)

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
        if getattr(self, 'save_recordings', False):
            # Create the recordings directory if it doesn't exist
            Path("recordings").mkdir(parents=True, exist_ok=True)

            filename = Path("recordings", f"recording_{self.client_id}_{int(time.time())}_audio.wav")
            save_recording(command.audio_bytes, filename)
            logger.info(f"Saved recording to {filename}")

        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(command.audio_bytes, dtype=NUMPY_AUDIO_FORMAT)

        # First use Whisper for STT
        transcription = self.whisper_engine.infer(np_audio, sampling_rate=AUDIO_SAMPLE_RATE)
        transcription = transcription.strip()

        logger.info(f"Transcription: {transcription}")

        self._do_text_inference(transcription)

    def _process_command_process_text(self, command):
        text = command.text

        self._do_text_inference(text)

    def _do_text_inference(self, text):
        # Use PydanticAgent for conversation/reasoning
        response = self.api.prompt_agent(self.client_id, text)

        logger.info(f"Response: {response}")
        
        spoken_response = response
        if not spoken_response:
            spoken_response = self.config.get(
                "nexus.client.no_response_error",
                "Something went wrong, there is no response.")
            
        # Use TTS engine to speak the response
        audio_tensor = self.tts_engine.infer(spoken_response, voice=self.config.tts.voice)

        # Convert the audio tensor to numpy array
        audio = self._tensor_to_int16(self._resample_audio(audio_tensor))
        
        # Play the response
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


            if is_speech or self._recording_state.is_recording():
                self._speech_buffer.append(audio_frames)

            if self._silence_duration > VAD_SILENCE_DURATION:
                if self._recording_state.is_recording():
                    if self._recording_state.is_confirmed():
                        logger.debug("Silence detected, stopping recording")
                        self.stopRecording()
                else:
                    logger.trace("Silence detected, clearing speech buffer")
                    self._speech_buffer.clear()


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

            detection: dict[str, float] = self.wake_word_model.predict(np_audio) # type: ignore

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

        transcription = self.whisper_engine.infer(np_audio, sampling_rate=AUDIO_SAMPLE_RATE)

        if getattr(self, 'save_recordings', False):
            # Create the recordings directory if it doesn't exist
            Path("recordings").mkdir(parents=True, exist_ok=True)
            
            # Save the recording to a file
            filename = Path("recordings", f"recording_{self.client_id}_{int(time.time())}_wakeword.wav")
            save_recording(command.audio_bytes, filename)
            logger.info(f"Saved recording to {filename}")

        transcription = transcription.strip()

        logger.debug(f"Transcription: {transcription}")

        model_conf = self.config.get("wake_word.models")
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
    
    def startRecording(self, confirmed=False):
        logger.debug("Recording started")
        self._recording_state.start()
        if confirmed:
            self._recording_state.confirm()

    def stopRecording(self, cancel=False):
        if cancel:
            logger.debug("Recording cancelled")
            self._recording_state.stop()
        elif not self._recording_state.is_confirmed:
            logger.trace("Recording not yet confirmed")
            return
        else:
            logger.debug("Recording stopped")
            self._recording_state.stop()
            audio_data = self._speech_buffer.get_bytes()
            logger.debug(f"Audio data length: {len(audio_data)}")

            if not audio_data:
                logger.warning("No audio data to process")
                return
            
            self.add_command(NexusVoiceClient.CommandProcessAudio(audio_data))

    def stop(self):
        if self.running:
            logger.info(f"Stopping {self.getName()}")
            self.running = False
            
            if self._recording_state.is_recording():
                self._speech_buffer.clear()
                self.stopRecording(cancel=True)

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
        def __init__(self, wake_word, audio_bytes: bytes):
            self.wake_word = wake_word
            self.audio_bytes = audio_bytes

    class CommandProcessAudio(Command):
        def __init__(self, audio_bytes):
            self.audio_bytes = audio_bytes

    class CommandProcessText(Command):
        def __init__(self, text):
            self.text = text

