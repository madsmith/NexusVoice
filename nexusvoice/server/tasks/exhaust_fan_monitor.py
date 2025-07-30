import asyncio
from datetime import datetime, timedelta
import logfire
import logging
from lutron_homeworks.commands import OutputCommand
from lutron_homeworks.commands.output import OutputAction
from typing import Any
from lutron_homeworks.database.types import LutronOutput

from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.core.config import NexusConfig
from nexusvoice.server.servivces.event_scheduler import EventSchedulerService, ScheduleEventIdT
from nexusvoice.server.servivces.lutron_homeworks import LutronHomeworksService

class ExhaustFanMonitor(NexusTask):
    required_services = {
        "event-scheduler": EventSchedulerService,
        "lutron-homeworks": LutronHomeworksService
    }

    def __init__(self, server: NexusServer, config: NexusConfig):
        super().__init__(server, config)

        self._schedule_ids: dict[int, ScheduleEventIdT] = {}

        self._lock = asyncio.Lock()

    async def start(self):
        """Run the task"""
        self.running = True
        
        lutron_service = self.get_required_service("lutron-homeworks", LutronHomeworksService)
        scheduler_service = self.get_required_service("event-scheduler", EventSchedulerService)

        # Get a list of all fan devices
        fan_devices = lutron_service.database().getOutputsByType("fan")
        exhaust_fans = [device for device in fan_devices if "exhaust" in device.name.lower()]
        fan_iids = [fan.iid for fan in exhaust_fans]

        def turn_off_fan(fan_iid: int):
            async def handler():
                async with self._lock:
                    # Remove the schedule if it still remains
                    # TODO: these logging statements are happening in the spans of other tasks
                    logfire.info(f"Turning off exhaust fan {fan_iid}")
                    self._schedule_ids.pop(fan_iid, None)
                    cmd = OutputCommand.set_zone_level(fan_iid, 0)
                    await lutron_service.execute_command(cmd)
            return handler

        async def on_output_command(event_data: list[Any]):
            fan_info: LutronOutput | None = next((fan for fan in exhaust_fans if fan.iid == event_data[0]), None)
            if fan_info is None:
                return

            assert fan_info is not None

            # Check is an exhaust fan was turned on
            if event_data[1] == OutputAction.ZONE_LEVEL.value:
                # On transition
                if event_data[2] > 0:
                    # TODO: these logging statements are happening in the spans of other tasks
                    logfire.info(f"Monitoring exhaust fan {fan_info.name}")
                    # Get delay configuration
                    fan_delay = self.config.get("exhaust_fan_monitor.fan_delay", 8 * 60)
                    if "shower" in fan_info.name.lower():
                        fan_delay = self.config.get("exhaust_fan_monitor.shower_fan_delay", 20 * 60)
                        
                    async with self._lock:
                        event_id = await scheduler_service.schedule_event(
                            "exhaust_fan_turn_off",
                            datetime.now() + timedelta(seconds=fan_delay),
                            turn_off_fan(fan_info.iid)
                        )
                        self._schedule_ids[fan_info.iid] = event_id

                else:
                    # Off transition
                    # TODO: these logging statements are happening in the spans of other tasks
                    logfire.info(f"Exhaust fan off: {fan_info.path}")
                    schedule_id = self._schedule_ids.pop(fan_info.iid, None)
                    if schedule_id is not None:
                        async with self._lock:
                            await scheduler_service.unschedule_event(schedule_id)

        # Monitor all output commands
        lutron_service.client().subscribe(OutputCommand, on_output_command)

        while self.running:
            await asyncio.sleep(1)
