import asyncio
import logfire
import logging
import numpy as np
from openwakeword.model import Model as OpenWakeWordModel
from pathlib import Path
import pyaudio
import silero_vad
from silero_vad.model import OnnxWrapper
import time
import torch
import torchaudio
from typing import Protocol

from nexusvoice.ai.AudioInferenceEngine import AudioInferenceEngine
from nexusvoice.ai.TTSInferenceEngine import TTSInferenceEngine
from nexusvoice.audio.utils import AudioBuffer, AudioData, save_recording, save_recording_async
from nexusvoice.audio.AudioDevice import AudioDevice
from nexusvoice.utils.RuntimeContextManager import RuntimeContextManager
from nexusvoice.core.api import NexusAPI, NexusAPIContext
from nexusvoice.core.api.online import NexusAPIOnline
from nexusvoice.core.config import NexusConfig
from nexusvoice.db.interaction_log import InteractionLog
from nexusvoice.utils.logging import get_logger
from nexusvoice.utils.debug import TimeThis

from nexusvoice.client.commands import (
    Command,
    CommandShutdown,
    CommandWakeWord,
    CommandProcessAudio,
    CommandProcessText
)

from nexusvoice.client.client import NexusClientBase
from nexusvoice.client.RecordingState import RecordingState, RecEvent, RecState

logger = get_logger(__name__)

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

class NexusVoiceClient:
    def __init__(self, client_id: str, config: NexusConfig, client: "NexusClientProtocol"):
        self.name = f"{self.__class__.__name__}::{client_id}"
        self.daemon = True

        self.client_id = client_id
        self.config = config

        self.client = client

        self._audio_device: AudioDevice | None = None

        self._wake_word_model: OpenWakeWordModel | None = None
        self._vad_model: OnnxWrapper | None = None
        self._whisper_engine: AudioInferenceEngine | None = None   
        self._tts_engine: TTSInferenceEngine | None = None
        
        self._command_queue = asyncio.Queue()

        self._wake_word_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        self._vad_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        self._speech_buffer: AudioBuffer = AudioBuffer(format=AUDIO_FORMAT)
        
        self._silence_duration = 0
        self._recording_state = RecordingState()

        self._interaction_log: InteractionLog = InteractionLog(
            self.config,
            Path(self.config.get("nexus.client.interaction_log_path", "logs"))
        )

        self._lock = asyncio.Lock()
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

    async def initialize(self):
        with logfire.span("Initialize"):
            await self.client.initialize(self)
            
            self._initialize_wake_word_model()
            self._initialize_VAD_model()
            self._initialize_whisper_model()
            self._initialize_TTS_model()

            self._initialize_audio_device()

            self._interaction_log.init_db()

    @logfire.instrument("Initialize Audio Device")
    def _initialize_audio_device(self):
        logger.info("Initializing audio device")

        self._audio_device = AudioDevice()
        self._audio_device.set_sample_delay(self.config.get("nexus.client.audio.sample_delay", -2200))

    @logfire.instrument("Initialize Wake Word Model")
    def _initialize_wake_word_model(self):
        logger.info("Initializing wake word model")

        model_conf = self.config.get("nexus.client.wake_word.models")
        if model_conf is None:
            logger.error("No wake word models found in config")
            raise RuntimeError("No wake word models found in config")

        models = [m.path if "path" in m else m.name for m in model_conf]
        
        self._wake_word_model = OpenWakeWordModel(
            wakeword_models=models,
            inference_framework=INFERENCE_FRAMEWORK,
            enable_speex_noise_suppression=True
        )

    @logfire.instrument("Initialize VAD Model")
    def _initialize_VAD_model(self):
        logger.info("Initializing VAD model")

        vad_model = silero_vad.load_silero_vad()
        if vad_model is None:
            logger.error("Failed to load VAD model")
            raise RuntimeError("Failed to load VAD model")
        self._vad_model = vad_model

    @logfire.instrument("Initialize Whisper Model")
    def _initialize_whisper_model(self):
        logger.info("Initializing Whisper STT...")

        model = self.config.get("nexus.whisper.processor.model", "openai/whisper-large-v2")
        self._whisper_engine = AudioInferenceEngine(model)
        self._whisper_engine.initialize()

    @logfire.instrument("Initialize TTS Model")
    def _initialize_TTS_model(self):
        logger.info("Initializing TTS model...")

        voice = self.config.get("nexus.tts.voice", None)
        self._tts_engine = TTSInferenceEngine(voices=voice)
        self._tts_engine.initialize()

    async def start(self):
        with logfire.span("start"):
            await self.client.start()

    async def run(self):
        try:
            logger.info(f"Starting {self.__class__.__name__} {self.client_id}")

            await self.initialize()
            await self.start()

            self.running = True

            command_processor_task = asyncio.create_task(
                self.process_commands(),
                name="Command Processor")
            audio_processing_task = asyncio.create_task(
                self.process_audio(),
                name="Mic Audio Processor")

            await asyncio.gather(command_processor_task, audio_processing_task)
        except asyncio.exceptions.CancelledError:
            pass

    async def process_audio(self):
        logger.info("Listening for audio")
        while self.running:
            try:
                wake_word_needed = WAKE_WORD_FRAME_CHUNK - self._wake_word_buffer.frame_count()
                vad_needed = SILERO_VAD_AUDIO_CHUNK - self._vad_buffer.frame_count()
                
                # Determine how much audio is needed to fill one of the buffers
                read_chunk_size = min(wake_word_needed, vad_needed)

                # TODO: async api on audio device
                audio_bytes = self.audio_device.read(read_chunk_size)

                self._wake_word_buffer.append(audio_bytes)
                self._vad_buffer.append(audio_bytes)

                self._process_vad()
                self._process_wake_word()

                # TODO: Can we get to the point where this is no longer necessary?
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                logger.info(f"Exiting due to KeyboardInterrupt - process_audio")
                break
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    # Show the traceback
                    import traceback
                    logger.error(traceback.format_exc())

        await self.stop()

    async def process_commands(self):
        """ Process commands from the command queue """
        command: Command | None = None
        try:
            async with asyncio.TaskGroup() as tg:
                while self.running:
                    try:
                        command = await self._command_queue.get()
                        tg.create_task(
                            self._process_command(command),
                            name="Process Command")
                    except asyncio.CancelledError:
                        logger.debug(f"CancelledError - process_commands")
                        logger.info(f"Shutting down {self.name}")
                        break
                    except KeyboardInterrupt:
                        logger.debug(f"Exiting due to KeyboardInterrupt - process_commands")
                        logger.info(f"Shutting down {self.name}")
                        break
                    except Exception as e:
                        logger.error(f"Error processing command: {e}")
                        # Show the traceback
                        import traceback
                        logger.error(traceback.format_exc())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            pass
        
        # await self.stop()
        logfire.info(f"Shutting down {self.name}")
        asyncio.create_task(
            self.stop(),
            name="Shutdown Client"
        )
    
    async def _process_command(self, command):
        try:
            logger.debug(f"Processing command {command}")
            if isinstance(command, CommandShutdown):
                logger.info(f"Shutting down {self.name}")
                await self.stop()
            elif isinstance(command, CommandWakeWord):
                await self._process_command_wake_word(command)
            elif isinstance(command, CommandProcessAudio):
                await self._process_command_process_audio(command)
            elif isinstance(command, CommandProcessText):
                await self._process_command_process_text(command)
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            raise
    
    async def _process_command_wake_word(self, command: CommandWakeWord):
        logger.debug(f"Received wake word command {command.wake_word}")
        if not self._confirm_wake_word(command):
            logger.debug(f"Wake word {command.wake_word} not confirmed")
            self.stopRecording(cancel=True)
        else:
            logger.trace(f"Wake word {command.wake_word} confirmed")
            audio_data = AudioData.from_wave("nexusvoice/client/resources/sounds/activation.wav")
            self.audio_device.play(audio_data)
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
        
    async def _process_command_process_audio(self, command: CommandProcessAudio):
        assert isinstance(command, CommandProcessAudio), "Command is not a process audio command"
        filename = Path("recordings", f"recording_{self.client_id}_{int(time.time())}_audio.wav")
        if self.config.get("nexus.client.save_recordings", False):
            task = asyncio.create_task(save_recording_async(command.audio_bytes, filename))
            def on_task_done(task: asyncio.Task[None]):
                try:
                    task.result()
                    logger.info(f"Saved recording to {filename}")
                except Exception as e:
                    logger.error(f"Error saving recording: {e}")
            task.add_done_callback(on_task_done)

        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(command.audio_bytes, dtype=NUMPY_AUDIO_FORMAT)

        # First use Whisper for STT
        transcription = self.whisper_engine.infer(np_audio, sampling_rate=AUDIO_SAMPLE_RATE)
        transcription = transcription.strip()

        if self.config.get("nexus.client.save_recordings", False):
            self._interaction_log.record_interaction(filename, transcription)

        logger.info(f"Transcription: {transcription}")

        await self._do_text_inference(transcription)

    async def _process_command_process_text(self, command: CommandProcessText):
        assert isinstance(command, CommandProcessText), "Command is not a process text command"
        text = command.text

        await self._do_text_inference(text)

    @logfire.instrument("Do Text Inference")
    async def _do_text_inference(self, text: str):
        response = await self.client.process_text(text)

        should_followup = response.endswith("?")

        logger.info(f"Response: {response}")
        
        spoken_response = response
        if not spoken_response:
            spoken_response = self.config.get(
                "nexus.client.no_response_error",
                "Something went wrong, there is no response.")
        
        await self.speak_response(spoken_response)
        
        if should_followup:
            self._silence_duration = 0
            self._recording_state.on_event(RecEvent.LISTEN)

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
                self._recording_state.on_event(RecEvent.VAD_DETECTED)
            else:
                self._silence_duration += SILERO_VAD_AUDIO_CHUNK / AUDIO_SAMPLE_RATE

            if is_speech or self._recording_state.is_recording():
                self._speech_buffer.append(audio_frames)

            if self._recording_state.state == RecState.SPEECH_PENDING:
                if self._silence_duration > self.config.get("nexus.audio.followup_timeout", 5):
                    logger.debug("Followup timeout reached, stopping recording")
                    self.stopRecording(cancel=True)

            if self._silence_duration > self.config.get("nexus.audio.speech_timeout", VAD_SILENCE_DURATION):
                if self._recording_state.is_recording():
                    if self._recording_state.is_processing_speech():
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

            # Discard frames if we are already recording
            if self._recording_state.is_recording():
                return

            # oww expects numpy array
            np_audio = np.frombuffer(audio_frames, dtype=NUMPY_AUDIO_FORMAT)

            detection: dict[str, float] = self.wake_word_model.predict(np_audio) # type: ignore

            if any(value > ACTIVATION_THRESHOLD for value in detection.values()):
                detected_wake_words = [key for key, value in detection.items() if value > ACTIVATION_THRESHOLD]
                logger.debug(f"\nWake word detected! {detected_wake_words}")
                self.wake_word_model.reset()
                for word in detected_wake_words:
                    print(f"Detected wake word: {word} VAD: {self._speech_buffer.get_duration_ms()}ms")
                    self.add_command(CommandWakeWord(word, self._speech_buffer.get_bytes()))
                    self.startRecording()

    async def speak_response(self, text: str):
        with logfire.span("TTS Inference"):
            audio_tensor = self.tts_engine.infer(text, voice=self.config.get("nexus.tts.voice"))

        # Convert the audio tensor to numpy array
        audio = self._tensor_to_int16(self._resample_audio(audio_tensor))

        # Play the response
        with logfire.span("Play Audio"):
            self.audio_device.play(audio)
            audio_data = AudioData(audio)
            await asyncio.sleep(audio_data.duration())

    def _confirm_wake_word(self, command: CommandWakeWord):
        assert isinstance(command, CommandWakeWord), "Command is not a wake word command"

        # TODO: Use Misaki to convert the wake word into phonemes and fuzzy match phonemes?

        # Convert audio bytes into numpy array
        np_audio = np.frombuffer(command.audio_bytes, dtype=NUMPY_AUDIO_FORMAT)

        transcription = self.whisper_engine.infer(np_audio, sampling_rate=AUDIO_SAMPLE_RATE)

        if self.config.get("nexus.client.save_recordings", False):
            # Save the recording to a file
            filename = Path("recordings", f"recording_{self.client_id}_{int(time.time())}_wakeword.wav")
            task: asyncio.Task[None] = asyncio.create_task(save_recording_async(command.audio_bytes, filename))
            def on_task_done(task: asyncio.Task[None]):
                try:
                    task.result()
                    self._interaction_log.record_wake_word(filename, command.wake_word, transcription)
                    logger.info(f"Saved recording to {filename}")
                except Exception as e:
                    logger.error(f"Error saving recording: {e}")
            task.add_done_callback(on_task_done)

        transcription = transcription.strip()

        logger.debug(f"Transcription: {transcription}")

        model_conf = self.config.get("nexus.client.wake_word.models")
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
        elif not self._recording_state.is_processing_speech():
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
            
            self.add_command(CommandProcessAudio(audio_data))

    async def stop(self):
        async with self._lock:
            if not self.running:
                return await asyncio.sleep(0)

                logger.info(f"Stopping {self.name}")
                self.running = False

                self.client.stop()

                if self._recording_state.is_recording():
                    self._speech_buffer.clear()
                    self.stopRecording(cancel=True)

                self.audio_device.shutdown()

    def add_command(self, command):
        logger.debug(f"Adding command {command}")
        self._command_queue.put_nowait(command)