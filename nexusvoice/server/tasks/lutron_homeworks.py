import asyncio
import logfire
import logging
from lutron_homeworks.client import LutronHomeworksClient
from lutron_homeworks.commands import LutronCommand
from lutron_homeworks.database import LutronDatabase, LutronXMLDataLoader
from lutron_homeworks.database.filters import FilterLibrary
from typing import Any

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.core.config import NexusConfig

from nexusvoice.server.servivces.lutron_homeworks import LutronHomeworksService

logger = logging.getLogger(__name__)

class LutronHomeworksImpl:
    def __init__(self, client: LutronHomeworksClient, database: LutronDatabase):
        self._client = client
        self._database = database
    
    def client(self) -> LutronHomeworksClient:
        return self._client
    
    def database(self) -> LutronDatabase:
        return self._database
    
    async def execute_command(self, command: LutronCommand) -> Any:
        try:
            return await self._client.execute_command(command)
        except Exception as e:
            logger.error(f"Error executing Lutron command: {e}")
            raise

class LutronHomeworks(NexusTask):

    provided_services = {
        "lutron-homeworks": LutronHomeworksService
    }

    def __init__(self, server: NexusServer, config: NexusConfig):
        super().__init__(server, config)

        host = config.get("lutron.server.host")
        username = config.get("lutron.server.username")
        password = config.get("lutron.server.password")

        self.lutron_client = LutronHomeworksClient(host, username, password)
        
        db_address = config.get("lutron.database.address")
        cache_path = config.get("lutron.database.cache_path", "./config/lutron")
        loader = LutronXMLDataLoader(db_address, cache_path)
        if config.get("lutron.database.cache_only"):
            # loader.set_cache_only(True)
            pass

        self.lutron_database = LutronDatabase(loader)

        # Apply database modifications
        # =================================================
        # 1. Define custom type map
        type_map = config.get("lutron.database.type_map")
        if type_map:
            self.lutron_database.apply_custom_type_map(type_map)

        # 2. Add data filters
        filters: dict[str, list[list[Any]]] = config.get("lutron.database.filters")
        for filter_name, instances in filters.items():
            for filter_args in instances:
                filter = FilterLibrary.get_filter(filter_name, filter_args)
                if filter is None:
                    raise RuntimeError(f"Filter {filter_name} not found")
                logger.debug(f"Applying filter {filter_name} with args {filter_args}")
                self.lutron_database.enable_filter(filter_name, filter_args)
        # =================================================

    async def _create_provided_services(self):
        self.lutron_homeworks_service = LutronHomeworksImpl(
            self.lutron_client,
            self.lutron_database
        )

        return {
            "lutron-homeworks": self.lutron_homeworks_service
        }
    
    async def start(self):
        """Run the task"""
        try:
            # Load the database
            with logfire.span("Load Lutron Database"):
                self.lutron_database.load()
            
            # Initialize the client
            with logfire.span("Initialize Lutron Client"):
                await self.lutron_client.connect()
            
            # TODO: need to be able to await connected.. connect isn't guaranteed to succeed

            self.running = True
            count = 1
            while self.running:
                # TODO: figure out what lutron_homeworks should do while running if anything
                # await self.server.broadcast(f"Hello from Lutron Homeworks {count}")
                count += 1
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Lutron Homeworks task failed: {e}")
            print("Exception: ", e, type(e))
            import traceback
            traceback.print_exc()
        finally:
            self.running = False