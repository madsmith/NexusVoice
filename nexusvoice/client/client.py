from typing import Protocol, TYPE_CHECKING
from nexusvoice.core.config import NexusConfig

if TYPE_CHECKING:
    from nexusvoice.client.voice_client import NexusVoiceClient
else:
    NexusVoiceClient = "NexusVoiceClient"

class NexusClientProtocol(Protocol):
    async def initialize(self, voice_client: NexusVoiceClient):
        ...

    async def start(self):
        ...

    async def stop(self):
        ...
        
    async def process_text(self, text: str) -> str:
        ...

class NexusClientBase(NexusClientProtocol):
    def __init__(self, client_id: str, config: NexusConfig):
        self.client_id = client_id
        self.config = config
