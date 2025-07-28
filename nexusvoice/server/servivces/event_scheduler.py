from typing import Protocol, runtime_checkable, Any
from datetime import datetime
from typing import Callable, Awaitable, Union

ScheduledEventCallbackT = Callable[[], Union[None, Awaitable[None]]]
ScheduleEventIdT = str

@runtime_checkable
class EventSchedulerService(Protocol):
    """Protocol defining the Event Scheduler service interface"""

    async def schedule_event(self, event_name: str, event_time: datetime, callback: ScheduledEventCallbackT) -> ScheduleEventIdT:
        ...

    async def unschedule_event(self, event_id: ScheduleEventIdT) -> None:
        ...