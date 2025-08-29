import logging
from enum import Enum
from typing import TypeVar, Generic

StateT = TypeVar("StateT", bound=Enum)
EventT = TypeVar("EventT", bound=Enum)

class StateMachine(Generic[StateT, EventT]):
    """
    Generic, typed state machine base class for Enum-based states and events.
    Subclass and specify StateT and EventT as Enum types for your domain.

    Example:
        class MyStates(Enum): ...
        class MyEvents(Enum): ...
        class MySM(StateMachine[MyStates, MyEvents]):
            TRANSITIONS = {
                MyStates.STOPPED: {MyEvents.START: MyStates.RUNNING},
                ...
            }
    """
    TRANSITIONS: dict[StateT, dict[EventT, StateT]] = {}

    def __init__(self, initial_state: StateT | None = None):
        if initial_state is not None:
            state = initial_state
        else:
            state = self._default_state()
        self.state: StateT = state

    def _default_state(self) -> StateT:
        # Returns the first value in the StateT enum, inferred from the first key in TRANSITIONS
        for key in self.TRANSITIONS.keys():
            enum_cls = type(key)
            return next(iter(enum_cls))
        raise NotImplementedError("State enum type could not be determined from TRANSITIONS. Please override _default_state().")

    def on_event(self, event: EventT) -> StateT:
        """
        Transition to the next state based on the current state and event.
        """
        if not isinstance(event, Enum):
            raise TypeError("EventT must be an Enum instance")
        transitions = self.TRANSITIONS.get(self.state, {})
        next_state = transitions.get(event)
        if next_state is None:
            logging.warning(f"Unhandled event {event} in state {self.state}")
        else:
            self.state = next_state
        return self.state

    def __repr__(self):
        return f"<{self.__class__.__name__} state={self.state}>"

    def __str__(self):
        return f"<{self.__class__.__name__} state={self.state}>"
