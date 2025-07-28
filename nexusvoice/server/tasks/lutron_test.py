import asyncio
import logfire
import random

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.core.config import NexusConfig
from nexusvoice.server.servivces.lutron_homeworks import LutronHomeworksService

class LutronTest(NexusTask):
    required_services = {
        "lutron-homeworks": LutronHomeworksService
    }

    async def start(self):
        """Run the task"""
        self.running = True

        lutron_service = self.get_required_service("lutron-homeworks", LutronHomeworksService)

        last_value = 0
        while self.running:
            await asyncio.sleep(300)

            try:
                # Pick a new value that is different from the last one
                while True:
                    value = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                    if value != last_value:
                        break

                from lutron_homeworks.commands import OutputCommand
                cmd = OutputCommand.set_zone_level(179, value)

                logfire.info(f"Setting output to {value}")

                await lutron_service.execute_command(cmd)
                last_value = value
            except Exception as e:
                logfire.error(f"Error executing Lutron command: {e}")


