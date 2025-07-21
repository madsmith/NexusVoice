import logfire
import pytest

from nexusvoice.core.config import load_config
from nexusvoice.client.RuntimeContextManager import RuntimeContextManager
from nexusvoice.core.api.base import AsyncContext
import asyncio

CONTEXT_TIMEOUT=2

class DummyContext(AsyncContext):
    _id_counter = 0
    def __init__(self):
        type(self)._id_counter += 1
        self.id = type(self)._id_counter
        logfire.info(f"DummyContext created: {self.id}")
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        await asyncio.sleep(.1)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        await asyncio.sleep(.05)

class SameTaskContext(DummyContext):
    def __init__(self):
        super().__init__()
        self._task = None

    async def __aenter__(self):
        await super().__aenter__()
        self._task = asyncio.current_task()
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await super().__aexit__(exc_type, exc, tb)
        exit_task = asyncio.current_task()
        if self._task != exit_task:
            raise Exception(f"__aexit__ called from a different task! Entered: {self._task}, Exited: {exit_task}")

class MockAPI:
    def __init__(self, context_provider):
        self.context_provider = context_provider
        self.context_created = False

    async def run_context(self):
        self.context_created = True
        return self.context_provider()

@pytest.fixture
def dummy_api():
    return MockAPI(lambda: DummyContext())

@pytest.fixture
def same_task_api():
    return MockAPI(lambda: SameTaskContext())

async def is_not_open(manager: RuntimeContextManager):
    context = manager.get_context("test")
    assert context is None, "Context should be removed"
    assert not manager._contexts["test"].is_open, "Context should be closed"
    assert manager._contexts["test"]._opened_at == -1, "Time should be unset when context is not open"

async def is_open(manager: RuntimeContextManager):
    context = manager.get_context("test")
    assert context is not None
    assert isinstance(context, DummyContext)
    assert manager._contexts["test"].is_open, "Context should be open"
    assert manager._contexts["test"]._opened_at != -1, "Time should be set when context is opened"

async def can_open_context(manager: RuntimeContextManager):
    await manager.open()
    
    await is_open(manager)

async def can_close_context(manager: RuntimeContextManager):
    await manager.close()

    await is_not_open(manager)

async def can_shutdown(manager: RuntimeContextManager):
    await manager.stop()
    try:
        task = manager.get_task()
        if task:
            await task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_runtime_context_manager_shutdown(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_shutdown_sloppy(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close_same_task(same_task_api):
    api = same_task_api

    manager = RuntimeContextManager()
    manager.add_context("test", same_task_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_open_open_close(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_close_from_another_task(same_task_api):
    api = same_task_api

    manager = RuntimeContextManager()
    manager.add_context("test", same_task_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await can_open_context(manager)

    # Close from a different task
    async def close_in_new_task():
        await manager.close()
    close_task = asyncio.create_task(close_in_new_task())
    await close_task  # Should NOT raise
    
    await is_not_open(manager)
    
    # Clean up
    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_reopen_after_close(dummy_api):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_context_expires(dummy_api):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    ctx = manager.get_context("test")
    assert ctx is not None
    assert isinstance(ctx, DummyContext)
    ctx_id = ctx.id

    await asyncio.sleep(CONTEXT_TIMEOUT - 1)

    ctx_still_open = manager.get_context("test")
    assert ctx_still_open is not None, "Context should still exist before timeout"

    await asyncio.sleep(2)

    ctx_later = manager.get_context("test")
    assert ctx_later is None, "Context should have expired"

    await can_open_context(manager)

    new_ctx = manager.get_context("test")
    assert new_ctx is not None, "Context should be reacquired after expiration"
    assert isinstance(new_ctx, DummyContext)
    assert new_ctx.id != ctx_id, "Context id should change after expiration"

    # await can_close_context(manager)
    await manager.close()
    
    await is_not_open(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_reopen_context_refreshes_timeout(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    ctx = manager.get_context("test")
    assert ctx is not None
    assert isinstance(ctx, DummyContext)
    ctx_id = ctx.id
    ctx_opened_at = manager._contexts["test"]._opened_at

    await asyncio.sleep(CONTEXT_TIMEOUT - 1)

    await can_open_context(manager)
    ctx_still_open = manager.get_context("test")
    assert ctx_still_open is not None, "Context should still exist before timeout"
    assert manager._contexts["test"]._opened_at is not None, "Context should have opened at"
    assert ctx_opened_at is not None, "Context should have opened at"
    
    reacquired_at = manager._contexts["test"]._opened_at
    assert ctx_opened_at < reacquired_at, "Context should have been reacquired"

    await asyncio.sleep(2)

    ctx_later = manager.get_context("test")
    assert ctx_later is not None, "Context should have been persisted"

    await asyncio.sleep(CONTEXT_TIMEOUT)

    ctx_expired = manager.get_context("test")
    assert ctx_expired is None, "Context should have expired"

    # await can_close_context(manager)
    await manager.close()
    
    await is_not_open(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_context_id_stable_over_time(dummy_api):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    ctx = manager.get_context("test")
    assert ctx is not None
    assert isinstance(ctx, DummyContext)
    ctx_id = ctx.id

    await asyncio.sleep(CONTEXT_TIMEOUT - 1)

    ctx_later = manager.get_context("test")
    assert ctx_later is not None, "Context should still exist before timeout"
    assert isinstance(ctx_later, DummyContext)

    assert ctx_later.id == ctx_id, f"Context id changed: was {ctx_id}, now {ctx_later.id}"
    await can_close_context(manager)

    await asyncio.sleep(2)

    ctx_expired = manager.get_context("test")
    assert ctx_expired is None, "Context should have expired"

    await can_open_context(manager)

    new_ctx = manager.get_context("test")
    assert new_ctx is not None, "Context should be reacquired after expiration"
    assert isinstance(new_ctx, DummyContext)
    assert new_ctx.id != ctx_id, "Context id should change after expiration"

    await can_close_context(manager)

    await can_shutdown(manager)
