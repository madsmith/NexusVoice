import asyncio
import logfire

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.core.config import NexusConfig

from nexusvoice.internal.api.server import NexusAPIServer

class AIAgent(NexusTask):
    def __init__(self, server: NexusServer, config: NexusConfig):
        super().__init__(server, config)
        self._api = NexusAPIServer(config)

    def register(self):
        self.server.register_command(
            "prompt_agent",
            self._prompt_agent,
            params={"prompt": str},
            description="Prompt an AI agent for inference"
        )

    async def start(self):
        """Run the task"""
        self.running = True

        await self._api.initialize()

        while self.running:
            await asyncio.sleep(1)

    async def _prompt_agent(self, client_id: str, prompt: str) -> str:
        """Handle sample command from client"""
        logfire.info(f"Prompting agent from {client_id}")
        self._api.attach_client(client_id)
        return await self._api.prompt_agent(client_id, prompt)

