import asyncio
import logfire
from nexusvoice.core.protocol.types import ClientInboundMessage
import numpy as np
import torch
import torchaudio

from nexusvoice.ai.TTSInferenceEngine import TTSInferenceEngine
from nexusvoice.audio import AudioDevice
from nexusvoice.audio.utils import AudioData
from nexusvoice.core.config import NexusConfig
from nexusvoice.core.protocol.connection import NexusConnection
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

class NexusAnnouncer:
    def __init__(self, host: str, port: int, client_id: str, config: NexusConfig):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._config = config

        self._running = False

        self._audio_device: AudioDevice | None = None
        self._tts_engine: TTSInferenceEngine | None = None
        self._nexus_connection: NexusConnection | None = None
        
    @property
    def audio_device(self) -> AudioDevice:
        assert self._audio_device is not None, f"{self.__class__.__name__} not initialized, audio device not available"
        return self._audio_device

    @property
    def tts_engine(self) -> TTSInferenceEngine:
        assert self._tts_engine is not None, f"{self.__class__.__name__} not initialized, tts engine not available"
        return self._tts_engine
    
    @property
    def nexus_connection(self) -> NexusConnection:
        assert self._nexus_connection is not None, f"{self.__class__.__name__} not initialized, nexus connection not available"
        return self._nexus_connection
    
    def initialize(self):
        with logfire.span("Initialize"):
            self._audio_device = AudioDevice()
            self._audio_device.set_mic_enabled(False)
            self._audio_device.initialize()
            self._audio_device.start()

            voice = self._config.get("nexus.tts.voice", None)
            self._tts_engine = TTSInferenceEngine(voices=voice)
            self._tts_engine.initialize()

            self._nexus_connection = NexusConnection(self._host, self._port)
            self._nexus_connection.on_server_message(self._handle_server_message)
    
    async def start(self):
        try:
            self._running = True
            await self.nexus_connection.connect()

            await self._announce("Announcer is online...")

            while self._running:
                await asyncio.sleep(0.1)

        except asyncio.exceptions.CancelledError:
            pass

    async def stop(self):
        try:
            self._running = False
            await self.nexus_connection.disconnect()
        except Exception as e:
            logger.exception("Nexus connection raised exception on shutdown")
        try:
            self.audio_device.stop()
        except Exception as e:
            logger.exception("Audio device raised exception on shutdown")

    async def _handle_server_message(self, message: str):
        logger.info(f"Received message: {message}")
        await self._announce(message)

    async def _announce(self, text: str):
        with logfire.span("TTS Inference"):
            audio_tensor = self.tts_engine.infer(text, voice=self._config.get("nexus.tts.voice"))

        # Convert the audio tensor to numpy array
        audio = self._tensor_to_int16(self._resample_audio(audio_tensor))

        # Play the response
        with logfire.span("Play Audio"):
            self.audio_device.play(audio)
            audio_data = AudioData(audio)
            await asyncio.sleep(audio_data.duration())

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
