import asyncio
from datetime import datetime, timedelta
import logfire
import uuid

from nexusvoice.core.config import NexusConfig
from nexusvoice.server.NexusServer import NexusServer
from nexusvoice.server.tasks.base import NexusTask
from nexusvoice.server.servivces.event_scheduler import ScheduledEventCallbackT, ScheduleEventIdT, EventSchedulerService


class ScheduledEvent:
    def __init__(self, event_name: str, event_time: datetime, callback: ScheduledEventCallbackT):
        self.event_name = event_name
        self.event_time = event_time
        self.callback: ScheduledEventCallbackT = callback

class Scheduler(NexusTask, EventSchedulerService):
    provided_services = { 
        "event-scheduler": EventSchedulerService 
    }

    def __init__(self, server: NexusServer, config: NexusConfig):
        super().__init__(server, config)
        self._scheduled_events: dict[ScheduleEventIdT, ScheduledEvent] = {}

        self._lock = asyncio.Lock()
        self._wait_event = asyncio.Event()

    def register(self):
        self.server.register_command(
            "schedule_notification", self._handle_schedule_notification,
            params={
                "message": str,
                "delay": int
            },
            description="Schedule a notification to be sent after a delay"
        )

        self.server.register_command(
            "unschedule_notification", self._handle_unschedule_notification,
            params={
                "event_id": str
            },
            description="Unschedule a notification"
        )

        self.server.register_command(
            "schedule_broadcast", self._handle_schedule_broadcast,
            params={
                "message": str,
                "delay": int
            },
            description="Schedule a broadcast to be sent after a delay"
        )
    
    async def _create_provided_services(self):
        return {
            "event-scheduler": self
        }    

    async def start(self):
        """Run the task"""
        self.running = True

        while self.running:
            now = datetime.now()
            next_event = None

            async with self._lock:
                # Iterate over a copy of the dicts keys and delete the events as they're processed
                for event_id in list(self._scheduled_events.keys()):
                    event = self._scheduled_events[event_id]
                    time_fmt = event.event_time.strftime("%Y-%m-%d %H:%M:%S")
                    logfire.trace(f"Processing event {event_id} at {time_fmt}")
                    if event.event_time <= now:
                        # Get task for invoking the callback
                        if asyncio.iscoroutinefunction(event.callback):
                            task = event.callback()
                        else:
                            task = asyncio.to_thread(event.callback)
                        
                        asyncio.create_task(
                            task,
                            name=f"scheduled-event_{event.event_name}")
                        
                        del self._scheduled_events[event_id]
                    elif next_event is None or event.event_time < next_event.event_time:
                        next_event = event
            
            if next_event is not None:
                await asyncio.sleep((next_event.event_time - now).total_seconds())
            else:
                # Wait for an event to be scheduled
                await self._wait_event.wait()
                self._wait_event.clear()
    
    async def schedule_event(self, event_name: str, event_time: datetime, callback: ScheduledEventCallbackT) -> ScheduleEventIdT:
        """Schedule an event"""
        async with self._lock:
            event_id = self._generate_event_id(event_name)
            self._scheduled_events[event_id] = ScheduledEvent(event_name, event_time, callback)
            self._wait_event.set()
            return event_id

    async def unschedule_event(self, event_id: ScheduleEventIdT) -> None:
        async with self._lock:
            if event_id in self._scheduled_events:
                del self._scheduled_events[event_id]

    def _generate_event_id(self, event_name: str) -> ScheduleEventIdT:
        return str(uuid.uuid4())
    
    # ==========================================================
    # Command handlers
    # ==========================================================
    async def _handle_schedule_notification(self, client_id: str, message: str, delay: int) -> ScheduleEventIdT:
        """Handle sample command from client"""
        logfire.info(f"Scheduled notification from {client_id}")
        
        async def send_response() -> None:
            await self.server.notify(client_id, message)

        event_id = await self.schedule_event("notification", datetime.now() + timedelta(seconds=delay), send_response)
        return event_id

    async def _handle_unschedule_notification(self, client_id: str, event_id: str) -> None:
        """Handle sample command from client"""
        logfire.info(f"Unscheduled notification from {client_id}")
        await self.unschedule_event(event_id)

    async def _handle_schedule_broadcast(self, client_id: str, message: str, delay: int) -> ScheduleEventIdT:
        """Handle sample command from client"""
        logfire.info(f"Scheduled broadcast from {client_id}")
        
        async def send_response() -> None:
            await self.server.broadcast(message)

        event_id = await self.schedule_event("broadcast", datetime.now() + timedelta(seconds=delay), send_response)
        return event_id