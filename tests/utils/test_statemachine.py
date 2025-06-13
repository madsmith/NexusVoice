from enum import Enum, auto
from nexusvoice.utils.state import StateMachine

class DummyStates(Enum):
    INIT = auto()
    RUNNING = auto()
    STOPPED = auto()

class DummyEvents(Enum):
    START = auto()
    STOP = auto()

class DummyStateMachine(StateMachine[DummyStates, DummyEvents]):
    TRANSITIONS = {
        DummyStates.INIT: {DummyEvents.START: DummyStates.RUNNING},
        DummyStates.RUNNING: {DummyEvents.STOP: DummyStates.STOPPED},
        DummyStates.STOPPED: {},
    }

def test_default_state():
    sm = DummyStateMachine()
    assert sm.state == DummyStates.INIT

def test_transition_start():
    sm = DummyStateMachine()
    sm.on_event(DummyEvents.START)
    assert sm.state == DummyStates.RUNNING

def test_transition_stop():
    sm = DummyStateMachine()
    sm.on_event(DummyEvents.START)
    sm.on_event(DummyEvents.STOP)
    assert sm.state == DummyStates.STOPPED

def test_unhandled_event():
    sm = DummyStateMachine()
    sm.on_event(DummyEvents.STOP)  # Should not change state
    assert sm.state == DummyStates.INIT
