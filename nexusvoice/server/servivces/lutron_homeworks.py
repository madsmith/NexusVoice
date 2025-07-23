from typing import Protocol, runtime_checkable, Any
from lutron_homeworks.client import LutronHomeworksClient
from lutron_homeworks.database import LutronDatabase
from lutron_homeworks.commands import LutronCommand

@runtime_checkable
class LutronHomeworksService(Protocol):
    """Protocol defining the Lutron service interface"""

    def client(self) -> LutronHomeworksClient: ...
    def database(self) -> LutronDatabase: ...
    
    async def execute_command(self, command: LutronCommand) -> Any:
        ...