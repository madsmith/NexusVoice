import logfire
import logging
from typing import TYPE_CHECKING

from nexusvoice.core.config import NexusConfig
from nexusvoice.client.client import NexusClientBase
from nexusvoice.core.api.online import NexusAPIOnline
from nexusvoice.server.connection import NexusConnection
from nexusvoice.utils.RuntimeContextManager import RuntimeContextManager

if TYPE_CHECKING:
    from nexusvoice.client.voice_client import NexusVoiceClient
else:
    NexusVoiceClient = "NexusVoiceClient"

logger = logging.getLogger(__name__)

class NexusClientServer(NexusClientBase):
    def __init__(self, host: str, port: int, client_id: str, config: NexusConfig):
        super().__init__(client_id, config)
        
        self._host = host
        self._port = port

        self._connection: NexusConnection = NexusConnection(self._host, self._port)

        self._voice_client: NexusVoiceClient | None = None

    @property
    def voice_client(self) -> NexusVoiceClient:
        if self._voice_client is None:
            raise RuntimeError("Voice client not initialized")
        return self._voice_client
    
    async def initialize(self, voice_client: NexusVoiceClient):
        self._voice_client = voice_client

    async def start(self):
        with logfire.span("Connection Lifecycle"):
            await self._connection.connect()

    async def stop(self):
        if self._connection.connected:
            await self._connection.disconnect()

    async def process_text(self, text: str):
        try:
            response = await self._connection.send_command(
                "prompt_agent", {"prompt": text})
            return response
        except Exception as e:
            # TODO: Remove This - api shouldn't throw exceptions
            logger.error(f"Error processing command: {e}")
            # Show the traceback
            import traceback
            logger.error(traceback.format_exc())
            return ""