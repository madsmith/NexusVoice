import threading

from enum import Enum, auto
from nexusvoice.utils.state import StateMachine
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

class RecState(Enum):
    STOPPED = auto()
    WAKE_PENDING = auto()
    SPEECH_PENDING = auto()
    ACTIVE_LISTEN = auto()
    PASSIVE_LISTEN = auto()

class RecEvent(Enum):
    START = auto()
    STOP = auto()
    CONFIRM = auto()
    LISTEN = auto()
    VAD_DETECTED = auto()

class RecordingState(StateMachine[RecState, RecEvent]):
    TRANSITIONS = {
        RecState.STOPPED: {
            RecEvent.START: RecState.WAKE_PENDING,
            RecEvent.VAD_DETECTED: RecState.STOPPED,
            RecEvent.LISTEN: RecState.SPEECH_PENDING,
        },
        RecState.WAKE_PENDING: {
            RecEvent.START: RecState.WAKE_PENDING,
            RecEvent.VAD_DETECTED: RecState.WAKE_PENDING,
            RecEvent.CONFIRM: RecState.ACTIVE_LISTEN,
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.SPEECH_PENDING: {
            RecEvent.VAD_DETECTED: RecState.PASSIVE_LISTEN,
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.ACTIVE_LISTEN: {
            RecEvent.VAD_DETECTED: RecState.ACTIVE_LISTEN,
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.PASSIVE_LISTEN: {
            RecEvent.VAD_DETECTED: RecState.PASSIVE_LISTEN,
            RecEvent.STOP: RecState.STOPPED,
        },
    }

    def __init__(self, initial_state=None):
        super().__init__(initial_state)
        self._lock = threading.Lock()

    def on_event(self, event: RecEvent):
        with self._lock:
            start_state = self.state
            super().on_event(event)
            logger.trace(f"RecordingState: {start_state} -> {self.state} [{event}]")
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
            self.state == RecState.WAKE_PENDING or
            self.state == RecState.SPEECH_PENDING
        )

    def is_processing_speech(self):
        return (
            self.state == RecState.ACTIVE_LISTEN or
            self.state == RecState.PASSIVE_LISTEN
        )

    def __str__(self):
        return self.state.name