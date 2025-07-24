from ast import TypeVar
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
import traceback
import logfire
from typing import Any, Awaitable, Callable, Generic, TypeVar, Union, cast

from nexusvoice.core.api.base import AsyncContext, NexusAPIContext
from nexusvoice.utils.state import StateMachine
from nexusvoice.utils.debug import AsyncRateLimiter

class ContextState(Enum):
    CLOSE_IDLE = auto()
    OPEN_IDLE = auto()
    SHUTTING_DOWN = auto()
    SHUT_DOWN = auto()
    NONE = auto()

    def __str__(self):
        return self.name

class ContextEvent(Enum):
    OPENED = auto()
    CLOSED = auto()
    SHUT_DOWN = auto()
    NONE = auto()

    def __str__(self):
        return self.name

class ContextManagerStateMachine(StateMachine):
    TRANSITIONS = {
        ContextState.CLOSE_IDLE: {
            ContextEvent.OPENED: ContextState.OPEN_IDLE,
            ContextEvent.CLOSED: ContextState.CLOSE_IDLE,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
        ContextState.OPEN_IDLE: {
            ContextEvent.OPENED: ContextState.OPEN_IDLE,
            ContextEvent.CLOSED: ContextState.CLOSE_IDLE,
            ContextEvent.SHUT_DOWN: ContextState.SHUTTING_DOWN,
        },
        ContextState.SHUTTING_DOWN: {
            ContextEvent.SHUT_DOWN: ContextState.SHUT_DOWN,
        },
    }

T = TypeVar("T")

class ManagedContext(Generic[T]):
    def __init__(
        self,
        context_provider: 
            Callable[[], AsyncContext[T]] | 
            Callable[[], Awaitable[AsyncContext[T]]],
        timeout: float
    ):
        self._context: AsyncContext[T] | None = None
        self._data: T | None = None
        self._context_provider = context_provider
        self._timeout = timeout
        self._opened_at = -1
        self._is_open = False

    @property
    def context(self) -> AsyncContext[T] | None:
        return self._context
    
    @property
    def data(self) -> T | None:
        return self._data

    @property
    def timeout(self) -> float:
        return self._timeout
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def close_at(self) -> float:
        return self._opened_at + self._timeout
    
    async def open(self):
        # Acquire the context if it's not already acquired
        if self._context is None:
            if asyncio.iscoroutinefunction(self._context_provider):
                self._context = await self._context_provider() 
            else:
                sync_provider = cast(Callable[[], AsyncContext[T]], self._context_provider)
                ctx: AsyncContext[T] = sync_provider() 
                self._context = ctx

        assert self._context is not None, "Context Provider failed to provide context"

        # Open the context
        self._data = await self._context.__aenter__()

        self._is_open = True
        self._opened_at = asyncio.get_event_loop().time()

    async def close(self):
        if self._context is None:
            return

        # Close the context
        await self._context.__aexit__(None, None, None)

        self._context = None
        self._data = None
        self._is_open = False
        self._opened_at = -1

    async def refresh(self):
        if self._context is None:
            return

        assert self._is_open, "Refresh against closed context"

        self._opened_at = asyncio.get_event_loop().time()

    def __repr__(self):
        return f"ManagedContext({self._context}, {self._data}, {self._is_open}, {self._opened_at}, {self._timeout})"

class RuntimeContextManager:
    def __init__(self):
        self._state_machine = ContextManagerStateMachine()

        self._contexts: dict[str, ManagedContext[Any]] = {}

        self._manager_task = None

        # Events for communication
        # Request events
        self._context_open_requested = asyncio.Event()
        self._context_close_requested = asyncio.Event()
        # Response events
        self._context_open_complete = asyncio.Event()
        self._context_close_complete = asyncio.Event()
        # Wake up event - called when request events are set
        self._context_wake_up_requested = asyncio.Event()

    def get_context(self, context_id: str) -> AsyncContext[Any] | None:
        return self._contexts[context_id].context

    def add_context(
        self,
        context_id: str,
        context_provider: Callable[[], AsyncContext] | Callable[[], Awaitable[AsyncContext]],
        timeout: float
    ):
        self._contexts[context_id] = ManagedContext(
            context_provider,
            timeout
        )
    
    async def open(self, context_ids: str | list[str] | None = None):
        """Request to open the context. Returns when context is open."""
        logfire.info("RuntimeContextManager::open")
        self._context_open_complete.clear()
        self._context_open_requested.set()
        self._context_wake_up_requested.set()
        await self._context_open_complete.wait()
        self._context_open_complete.clear()

    async def close(self, context_ids: str | list[str] | None = None):
        """Request to close the context. Returns when context is closed."""
        logfire.info("RuntimeContextManager::close")
        self._context_close_complete.clear()
        self._context_close_requested.set()
        self._context_wake_up_requested.set()
        await self._context_close_complete.wait()
        self._context_close_complete.clear()

    def start(self):
        if self._manager_task is None or self._manager_task.done():
            self._manager_task = asyncio.create_task(
                self._context_manager_multi(),
                name="RuntimeContextManager",
            )

    async def stop(self):
        if self._manager_task:
            logfire.info("RuntimeContextManager::stop")
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                pass

    def get_task(self):
        return self._manager_task

    async def _context_manager_multi(self):
        rate_limiter = AsyncRateLimiter(100, per_seconds=20)
        while True:
            try:
                await rate_limiter.acquire()
                if self._state_machine.state == ContextState.SHUT_DOWN:
                    logfire.info("ContextManager: Shut down")
                    break
                elif self._state_machine.state == ContextState.SHUTTING_DOWN:
                    logfire.info("ContextManager: Shutting down")
                    self._context_close_requested.set()
                    self._context_wake_up_requested.set()
                else:
                    timeout_info: tuple[str, float] | None = None
                    try:
                        # Check if any contexts need to timeout
                        timeout_info = self._get_next_timeout()
                        if timeout_info is not None:
                            logfire.info(f"ContextManager: Waiting for timeout on context {timeout_info[0]} ({timeout_info[1]:.1f}s)")
                            await asyncio.wait_for(self._context_wake_up_requested.wait(), timeout_info[1])
                        else:
                            logfire.info("ContextManager: Waiting for next event")
                            await self._context_wake_up_requested.wait()
                    except asyncio.TimeoutError:
                        if timeout_info is None:
                            logfire.info("ContextManager: No timeout info")
                            continue
                        logfire.info(f"ContextManager: Context Timeout {timeout_info[0]}")
                        await self._close_context(timeout_info[0])
                        continue
                
                    # Wake up must have occurred - clear event
                    self._context_wake_up_requested.clear()
                
                # Handle open / close events
                if self._context_open_requested.is_set():
                    logfire.info(f"ContextManager: Open requested")
                    await self._open_contexts()
                    self._context_open_requested.clear()
                    self._context_open_complete.set()
                    self._state_machine.on_event(ContextEvent.OPENED)

                elif self._context_close_requested.is_set():
                    logfire.info(f"ContextManager: Close requested")
                    await self._close_contexts()
                    self._context_close_requested.clear()
                    self._context_close_complete.set()
                    is_shutdown = self._state_machine.state == ContextState.SHUTTING_DOWN
                    event = ContextEvent.SHUT_DOWN if is_shutdown else ContextEvent.CLOSED
                    self._state_machine.on_event(event)
                else:
                    logfire.warning("ContextManager: Unknown wake up event cause")
            
            except asyncio.CancelledError:
                logfire.info("ContextManager: Cancelled")
                if self._state_machine.state != ContextState.SHUT_DOWN:
                    self._state_machine.on_event(ContextEvent.SHUT_DOWN)
            except Exception as e:
                logfire.error(f"ContextManager: Exception: {e} {[type(e).__name__]}")
                if self._state_machine.state != ContextState.SHUT_DOWN:
                    self._state_machine.on_event(ContextEvent.SHUT_DOWN)
            except GeneratorExit:
                self._state_machine.on_event(ContextEvent.SHUT_DOWN)
                raise
            except BaseException as e:
                logfire.error(f"ContextManager: BaseException: {e} {[type(e).__name__]}")
                # import traceback
                # logfire.error(traceback.format_exc())
                if self._state_machine.state != ContextState.SHUT_DOWN:
                    self._state_machine.on_event(ContextEvent.SHUT_DOWN)
    
    def _get_next_timeout(self) -> tuple[str, float] | None:
        timeouts = []
        now = asyncio.get_event_loop().time()
        for context_id, managed_context in self._contexts.items():
            if managed_context.is_open:
                # Compute the time until the context expires
                close_at = managed_context.close_at
                timeout = close_at - now
                timeouts.append((context_id, timeout))
        if timeouts:
            return min(timeouts, key=lambda x: x[1])
        return None
    
    async def _open_contexts(self):
        for context_id, managed_context in self._contexts.items():
            if not managed_context.is_open:
                logfire.info(f"ContextManager: Opening context for {context_id}")
                await managed_context.open()
            else:
                logfire.info(f"ContextManager: Refreshing context for {context_id}")
                await managed_context.refresh()
    
    async def _close_contexts(self):
        for context_id, managed_context in self._contexts.items():
            if managed_context.is_open:
                logfire.info(f"ContextManager: Closing context for {context_id}")
                await managed_context.close()

    async def _close_context(self, context_id: str):
        managed_context = self._contexts.get(context_id)
        if managed_context is None:
            logfire.warning(f"ContextManager: No context found for {context_id}")
            return
        if managed_context.is_open:
            logfire.info(f"ContextManager: Closing context for {context_id}")
            await managed_context.close()
