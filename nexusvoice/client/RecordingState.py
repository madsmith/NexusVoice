import threading

from enum import Enum, auto
from nexusvoice.utils.state import StateMachine

class RecState(Enum):
    STOPPED = auto()
    PENDING = auto()
    ACTIVE_LISTEN = auto()
    PASSIVE_LISTEN = auto()

class RecEvent(Enum):
    START = auto()
    STOP = auto()
    CONFIRM = auto()
    LISTEN = auto()

class RecordingState(StateMachine[RecState, RecEvent]):
    TRANSITIONS = {
        RecState.STOPPED: {
            RecEvent.START: RecState.PENDING,
            RecEvent.LISTEN: RecState.PASSIVE_LISTEN,
        },
        RecState.PENDING: {
            RecEvent.CONFIRM: RecState.ACTIVE_LISTEN,
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.ACTIVE_LISTEN: {
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.PASSIVE_LISTEN: {
            RecEvent.STOP: RecState.STOPPED,
        },
    }

    def __init__(self, initial_state=None):
        super().__init__(initial_state)
        self._lock = threading.Lock()

    def on_event(self, event: RecEvent):
        with self._lock:
            super().on_event(event)
            return self.state

    def start(self):
        self.on_event(RecEvent.START)

    def stop(self):
        self.on_event(RecEvent.STOP)

    def confirm(self):
        self.on_event(RecEvent.CONFIRM)

    def listen(self):
        self.on_event(RecEvent.LISTEN)

    def is_recording(self):
        return (
            self.state == RecState.ACTIVE_LISTEN or 
            self.state == RecState.PASSIVE_LISTEN or
            self.state == RecState.PENDING 
        )

    def is_processing_speech(self):
        return (
            self.state == RecState.ACTIVE_LISTEN or
            self.state == RecState.PASSIVE_LISTEN
        )

    def __str__(self):
        return self.state.name