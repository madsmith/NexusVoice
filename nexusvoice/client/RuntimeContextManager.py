import asyncio
import logfire
from typing import Optional

from nexusvoice.utils.debug import TimeThis

from enum import Enum, auto

class ContextState(Enum):
    CLOSE_IDLE = auto()
    OPEN_IDLE = auto()
    OPENING = auto()
    CLOSING = auto()
    REFRESH = auto()
    SHUTTING_DOWN = auto()
    SHUT_DOWN = auto()
    NONE = auto()

    def __str__(self):
        return self.name

class ContextEvent(Enum):
    OPEN_REQUESTED = auto()
    CLOSE_REQUESTED = auto()
    CLOSE_TIMEOUT = auto()
    OPENED = auto()
    CLOSED = auto()
    SHUT_DOWN = auto()
    NONE = auto()

    def __str__(self):
        return self.name

class ContextStateMachine:
    # State transition table: {current_state: {event: next_state}}
    TRANSITIONS = {
        ContextState.CLOSE_IDLE: {
            ContextEvent.OPEN_REQUESTED: ContextState.OPENING,
            ContextEvent.CLOSE_REQUESTED: ContextState.CLOSING,
            ContextEvent.SHUT_DOWN: ContextState.SHUT_DOWN,
        },
        ContextState.OPENING: {
            ContextEvent.OPENED: ContextState.OPEN_IDLE,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
        ContextState.OPEN_IDLE: {
            ContextEvent.CLOSE_REQUESTED: ContextState.CLOSING,
            ContextEvent.CLOSE_TIMEOUT: ContextState.CLOSING,
            ContextEvent.OPEN_REQUESTED: ContextState.REFRESH,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
        ContextState.CLOSING: {
            ContextEvent.CLOSED: ContextState.CLOSE_IDLE,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
        ContextState.REFRESH: {
            ContextEvent.OPENED: ContextState.OPEN_IDLE,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
    }

    def __init__(self):
        self.state: ContextState = ContextState.CLOSE_IDLE

    def on_event(self, event: Optional[ContextEvent]) -> ContextState:
        if event is not None:
            transitions = self.TRANSITIONS.get(self.state, {})
            next_state = transitions.get(event, ContextState.NONE)
            if next_state != ContextState.NONE:
                logfire.info(f"ContextStateMachine: {self.state} -> {next_state} (event: {event})")
                self.state = next_state
        return self.state

    def __repr__(self):
        return f"<ContextStateMachine state={self.state}>"

class RuntimeContextManager:
    def __init__(self, api, context_timeout=15):
        self.api = api
        self.context_timeout = context_timeout

        self._state_machine = ContextStateMachine()

        self._context = None
        self._context_open = False
        self._context_opened_at = -1
        self._manager_task = None

        # Events for communication
        self._context_open_requested = asyncio.Event()
        self._context_close_requested = asyncio.Event()
        self._context_open_complete = asyncio.Event()
        self._context_close_complete = asyncio.Event()
        self._context_wake_up_requested = asyncio.Event()

    def get_context(self):
        return self._context

    async def open(self):
        """Request to open the context. Returns when context is open."""
        logfire.info("RuntimeContextManager::open")
        with TimeThis("Context Open Request"):
            self._context_open_complete.clear()
            self._context_open_requested.set()
            self._context_wake_up_requested.set()
            logfire.info("RuntimeContextManager::open: Requested Open")
            await self._context_open_complete.wait()
        self._context_open_complete.clear()

    async def close(self):
        """Request to close the context. Returns when context is closed."""
        logfire.info("RuntimeContextManager::close")
        with TimeThis("Context Close Request"):
            self._context_close_complete.clear()
            self._context_close_requested.set()
            self._context_wake_up_requested.set()
            await self._context_close_complete.wait()
        self._context_close_complete.clear()

    def start(self):
        if self._manager_task is None or self._manager_task.done():
            self._manager_task = asyncio.create_task(
                self._context_manager(),
                name="RuntimeContextManager",
            )

    async def stop(self):
        if self._manager_task:
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                pass

    def get_task(self):
        return self._manager_task

    async def _open_context(self):
        if self._context:
            with TimeThis("Internal Context Open"):
                await self._context.__aenter__()

            self._context_open = True
            self._context_opened_at = asyncio.get_event_loop().time()
            self._context_open_complete.set()

    async def _refresh_context(self):
        assert self._context is not None
        self._context_open = True
        self._context_opened_at = asyncio.get_event_loop().time()
        self._context_open_complete.set()

    async def _close_context(self):
        if self._context:
            with TimeThis("Internal Context Close"):
                await self._context.__aexit__(None, None, None)

            self._context = None
            self._context_open = False
            self._context_opened_at = -1
            self._context_close_complete.set()

    def _map_event(self) -> Optional[ContextEvent]:
        if self._context_open_requested.is_set():
            return ContextEvent.OPEN_REQUESTED
        elif self._context_close_requested.is_set():
            return ContextEvent.CLOSE_REQUESTED
        return None
    
    async def _context_manager(self):
        while True:
            try:
                state = self._state_machine.state
                logfire.info(f"ContextManager: State: {state}")

                if state == ContextState.CLOSE_IDLE or state == ContextState.OPEN_IDLE:
                    event: Optional[ContextEvent] = None
                    if state == ContextState.CLOSE_IDLE:
                        logfire.info("ContextManager: Waiting for open")
                        await self._context_wake_up_requested.wait()
                    else:
                        logfire.info(f"ContextManager: Waiting for close or timeout ({self.context_timeout}s)")
                        try:
                            await asyncio.wait_for(self._context_wake_up_requested.wait(), timeout=self.context_timeout)
                        except asyncio.TimeoutError:
                            logfire.info("ContextManager: Timeout")
                            event = ContextEvent.CLOSE_TIMEOUT

                    event = event if event else self._map_event()
                    logfire.info(f"ContextManager: Event: {event}")

                    self._context_open_requested.clear()
                    self._context_close_requested.clear()
                    self._context_wake_up_requested.clear()

                    self._state_machine.on_event(event)

                elif state == ContextState.OPENING:
                    self._context = await self.api.run_context()
                    await self._open_context()
                    self._state_machine.on_event(ContextEvent.OPENED)
                    self._context_open_complete.set()

                elif state == ContextState.CLOSING or state == ContextState.SHUTTING_DOWN:
                    await self._close_context()

                    event = ContextEvent.SHUT_DOWN if state == ContextState.SHUTTING_DOWN else ContextEvent.CLOSED
                    self._state_machine.on_event(event)
                    self._context_close_complete.set()

                elif state == ContextState.REFRESH:
                    await self._refresh_context()
                    self._state_machine.on_event(ContextEvent.OPENED)
                    self._context_open_complete.set()

                elif state == ContextState.SHUT_DOWN:
                    # Terminate the loop
                    break

                else:
                    raise Exception(f"Unknown state: {state}")

            except asyncio.CancelledError:
                logfire.info("ContextManager: Cancelled")
                self._state_machine.on_event(ContextEvent.SHUT_DOWN)
            except Exception as e:
                logfire.error(f"ContextManager: Exception: {e}")
                self._state_machine.on_event(ContextEvent.SHUT_DOWN)
