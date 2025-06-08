import asyncio
import logfire

class RuntimeContextManager:
    def __init__(self, api, context_timeout=15):
        self.api = api
        self.context_timeout = context_timeout

        self._context = None
        self._context_open = False
        self._context_opened_at = None
        self._manager_task = None

        # Events for communication
        self._context_open_requested = asyncio.Event()
        self._context_close_requested = asyncio.Event()
        self._context_open_complete = asyncio.Event()
        self._context_close_complete = asyncio.Event()

    def get_context(self):
        return self._context

    async def open(self):
        """Request to open the context. Returns when context is open."""
        logfire.info("Context opening due to open request")
        self._context_open_requested.set()
        await self._context_open_complete.wait()
        self._context_open_complete.clear()

    async def close(self):
        """Request to close the context. Returns when context is closed."""
        logfire.info("Context closing due to close request")
        self._context_close_requested.set()
        await self._context_close_complete.wait()
        self._context_close_complete.clear()

    def start(self):
        if self._manager_task is None or self._manager_task.done():
            self._manager_task = asyncio.create_task(
                self._context_manager(),
                name="RuntimeContextManager",
            )

    def stop(self):
        if self._manager_task:
            self._manager_task.cancel()

    def get_task(self):
        return self._manager_task

    async def _context_manager(self):
        try:
            while True:
                logfire.info("Context manager waiting for open request")
                await self._context_open_requested.wait()
                self._context_open_requested.clear()

                with logfire.span("Context Manager Lifecycle"):

                    logfire.info("Acquiring context")
                    self._context = await self.api.run_context()
                    logfire.info("Context acquired")
                    try:
                        with logfire.span("Context Open"):
                            await self._context.__aenter__()

                            self._context_open = True
                            self._context_opened_at = asyncio.get_event_loop().time()
                            self._context_open_complete.set()

                        with logfire.span("Context Held Open"):
                            try:
                                await asyncio.wait_for(self._context_close_requested.wait(), timeout=self.context_timeout)
                                # Stop requested
                                logfire.info("Held released for close")
                            except asyncio.TimeoutError:
                                logfire.warning("Context held open for too long, closing")
                            finally:
                                self._context_close_requested.clear()
                    finally:
                        with logfire.span("Context Closed"):
                            if self._context:
                                logfire.info("Context closing due to close request")
                                await self._context.__aexit__(None, None, None)
                                self._context = None
                                self._context_open = False
                                self._context_opened_at = None
                                self._context_close_complete.set()
                                logfire.info(f"Context closed: {self._context}")
        
        except asyncio.CancelledError:
            logfire.info("RuntimeContextManager task cancelled, shutting down.")
            # Cleanup if needed
            if self._context:
                await self._context.__aexit__(None, None, None)
                self._context = None
                self._context_open = False
                self._context_opened_at = None
                self._context_close_complete.set()
            raise
