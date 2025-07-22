from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexusvoice.server.NexusServer import NexusServer

class NexusTask(ABC):
    """Base class for all Nexus tasks."""
    
    def __init__(self, server: "NexusServer"):
        self.server = server
        self.running = False

    def register(self):
        """
        Register the task with the server.
        """
        pass

    async def start(self):
        pass

    async def stop(self):
        self.running = False
        # TODO: Implement proper cleanup of running async task if exists