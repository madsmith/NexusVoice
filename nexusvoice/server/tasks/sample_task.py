import asyncio
import logfire

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.types import BroadcastMessage
from nexusvoice.server.tasks.base import NexusTask

class SampleTask(NexusTask):
    def __init__(self, server: "NexusServer"):
        super().__init__(server)

    def register(self):
        self.server.register_command("sample", self._handle_sample)

    async def start(self):
        """Run the task"""
        self.running = True
        count = 1
        while self.running:
            await asyncio.sleep(5)
            await self.server.broadcast(BroadcastMessage(message=f"Hello from SampleTask {count}"))
            count += 1

    async def _handle_sample(self, client_id: str, payload: dict) -> str:
        """Handle sample command from client"""
        logfire.info(f"Sample from {client_id}")
        return "no sample for you"

