import logfire
import logging
from typing import TYPE_CHECKING

from nexusvoice.core.config import NexusConfig
from nexusvoice.client.client import NexusClientBase
from nexusvoice.internal.api.online import NexusAPIOnline
from nexusvoice.utils.RuntimeContextManager import RuntimeContextManager

if TYPE_CHECKING:
    from nexusvoice.client.voice_client import NexusVoiceClient
else:
    NexusVoiceClient = "NexusVoiceClient"

logger = logging.getLogger(__name__)

class NexusClientStandalone(NexusClientBase):
    def __init__(self, client_id: str, config: NexusConfig):
        super().__init__(client_id, config)
        self._api: NexusAPIOnline = NexusAPIOnline(self.config)
        self._context_manager: RuntimeContextManager = RuntimeContextManager()

        self._voice_client: NexusVoiceClient | None = None

    @property
    def voice_client(self) -> NexusVoiceClient:
        if self._voice_client is None:
            raise RuntimeError("Voice client not initialized")
        return self._voice_client
    
    async def initialize(self, voice_client: NexusVoiceClient):
        self._voice_client = voice_client

        await self._api.initialize()

        # Initialize Context Manager
        agent_context_provider = self._api.create_api_context
        history_context_provider = self._api.create_history_context

        self._context_manager.add_context(
            "agent-context",
            agent_context_provider,
            self.config.get("nexus.client.timeouts.agent_context", 300)
        )

        self._context_manager.add_context(
            "history-context",
            history_context_provider,
            self.config.get("nexus.client.timeouts.history_context", 60)
        )

    async def start(self):
        with logfire.span("ContextManager Lifecycle"):
            self._context_manager.start()

    async def stop(self):
        await self._context_manager.close()
        await self._context_manager.stop()

    async def process_text(self, text: str):
        await self._context_manager.open()

        try:
            response = await self._api.prompt_agent(self.client_id, text)
            return response
        except Exception as e:
            # TODO: Remove This - api shouldn't throw exceptions
            logger.error(f"Error processing command: {e}")
            # Show the traceback
            import traceback
            logger.error(traceback.format_exc())
            return ""