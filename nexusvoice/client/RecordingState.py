import threading

from enum import Enum, auto
from nexusvoice.utils.state import StateMachine

class RecState(Enum):
    STOPPED = auto()
    PENDING = auto()
    RECORDING = auto()

class RecEvent(Enum):
    START = auto()
    STOP = auto()
    CONFIRM = auto()

class RecordingState(StateMachine[RecState, RecEvent]):
    TRANSITIONS = {
        RecState.STOPPED: {
            RecEvent.START: RecState.PENDING,
        },
        RecState.PENDING: {
            RecEvent.CONFIRM: RecState.RECORDING,
            RecEvent.STOP: RecState.STOPPED,
        },
        RecState.RECORDING: {
            RecEvent.STOP: RecState.STOPPED,
        },
    }

    def __init__(self, initial_state=None):
        super().__init__(initial_state)
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.on_event(RecEvent.START)

    def stop(self):
        with self._lock:
            self.on_event(RecEvent.STOP)

    def confirm(self):
        with self._lock:
            self.on_event(RecEvent.CONFIRM)

    def is_recording(self):
        with self._lock:
            return self.state == RecState.RECORDING or self.state == RecState.PENDING

    def is_confirmed(self):
        with self._lock:
            return self.state == RecState.RECORDING

    def __str__(self):
        return self.state.name