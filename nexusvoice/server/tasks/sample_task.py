import asyncio
import logfire

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.core.config import NexusConfig

class SampleTask(NexusTask):
    def __init__(self, server: NexusServer, config: NexusConfig):
        super().__init__(server, config)

    def register(self):
        self.server.register_command("sample", self._handle_sample)

    async def start(self):
        """Run the task"""
        self.running = True
        count = 1
        while self.running:
            # await self.server.broadcast(f"Hello from SampleTask {count}")
            count += 1
            await asyncio.sleep(30)

    async def _handle_sample(self, client_id: str) -> str:
        """Handle sample command from client"""
        logfire.info(f"Sample from {client_id}")
        return "no sample for you"

