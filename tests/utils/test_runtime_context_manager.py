import asyncio
import logfire
import pytest

from nexusvoice.utils.RuntimeContextManager import RuntimeContextManager
from nexusvoice.internal.api.base import AsyncContext

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
def dummy_api() -> MockAPI:
    return MockAPI(lambda: DummyContext())

@pytest.fixture
def same_task_api() -> MockAPI:
    return MockAPI(lambda: SameTaskContext())

async def is_not_open(manager: RuntimeContextManager, context_ids: list[str]):
    for context_id in context_ids:
        context = manager.get_context(context_id)
        assert context is None, f"Context {context_id} should be removed"
        assert not manager._contexts[context_id].is_open, f"Context {context_id} should be closed"
        assert manager._contexts[context_id]._opened_at == -1, f"Time should be unset when context {context_id} is not open"

async def is_open(manager: RuntimeContextManager, context_ids: list[str]):
    for context_id in context_ids:
        context = manager.get_context(context_id)
        assert context is not None
        assert isinstance(context, DummyContext)
        assert manager._contexts[context_id].is_open, f"Context {context_id} should be open"
        assert manager._contexts[context_id]._opened_at != -1, f"Time should be set when context {context_id} is opened"

async def can_open_context(manager: RuntimeContextManager, context_ids: list[str] | None = None):
    await manager.open(context_ids)
    
    if context_ids is None:
        context_ids = ["test"]
    
    await is_open(manager, context_ids)

async def can_close_context(manager: RuntimeContextManager, context_ids: list[str] | None = None):
    await manager.close(context_ids)
    
    if context_ids is None:
        context_ids = ["test"]
    
    await is_not_open(manager, context_ids)

async def can_shutdown(manager: RuntimeContextManager):
    await manager.stop()
    try:
        task = manager.get_task()
        if task:
            await task
    except asyncio.CancelledError:
        pass

async def can_get_context_id(manager: RuntimeContextManager, context_id: str):
    context = manager.get_context(context_id)
    assert context is not None, f"Context {context_id} should exist"
    assert isinstance(context, DummyContext), f"Context {context_id} should be a DummyContext"
    return context.id

@pytest.mark.asyncio
async def test_runtime_context_manager_shutdown(dummy_api: MockAPI):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_shutdown_sloppy(dummy_api: MockAPI):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close(dummy_api: MockAPI):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close_same_task(same_task_api: MockAPI):
    api = same_task_api

    manager = RuntimeContextManager()
    manager.add_context("test", same_task_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_open_open_close(dummy_api: MockAPI):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_close_from_another_task(same_task_api: MockAPI):
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
    
    await is_not_open(manager, ["test"])
    
    # Clean up
    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_reopen_after_close(dummy_api: MockAPI):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_context_expires(dummy_api: MockAPI):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    ctx_id = await can_get_context_id(manager, "test")

    await asyncio.sleep(CONTEXT_TIMEOUT - 1)

    ctx_still_open = manager.get_context("test")
    assert ctx_still_open is not None, "Context should still exist before timeout"

    await asyncio.sleep(2)

    ctx_later = manager.get_context("test")
    assert ctx_later is None, "Context should have expired"

    await can_open_context(manager)

    new_ctx_id = await can_get_context_id(manager, "test")
    assert new_ctx_id != ctx_id, "Context id should change after expiration"

    # await can_close_context(manager)
    await manager.close()
    
    await is_not_open(manager, ["test"])

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_reopen_context_refreshes_timeout(dummy_api: MockAPI):
    api = dummy_api
    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    ctx_id = await can_get_context_id(manager, "test")
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
    
    await is_not_open(manager, ["test"])

    await can_shutdown(manager)

@pytest.mark.asyncio
async def test_context_manager_context_id_stable_over_time(dummy_api: MockAPI):
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    await is_not_open(manager, ["test"])

    await can_open_context(manager)

    ctx_id = await can_get_context_id(manager, "test")

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

    new_ctx_id = await can_get_context_id(manager, "test")
    assert new_ctx_id != ctx_id, "Context id should change after expiration"

    await can_close_context(manager)

    await can_shutdown(manager)


@pytest.mark.asyncio
async def test_context_manager_open_by_specific_id(dummy_api: MockAPI):
    """Test that we can open a context by specific ID"""
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test1", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.add_context("test2", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    # Both contexts should be initially closed
    await is_not_open(manager, ["test1", "test2"])

    # Open just the first context
    await can_open_context(manager, ["test1"])
    
    # First context should be open now
    await is_open(manager, ["test1"])
    
    # Second context should still be closed
    await is_not_open(manager, ["test2"])
    
    # Second context should still be closed
    await is_not_open(manager, ["test2"])
    
    # Close all contexts
    await can_close_context(manager, ["test1", "test2"])
    
    # Both contexts should be closed now
    await is_not_open(manager, ["test1", "test2"])

    await can_shutdown(manager)


@pytest.mark.asyncio
async def test_context_manager_multiple_contexts_with_different_ids(dummy_api: MockAPI):
    """Test managing multiple contexts with different IDs"""
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test1", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.add_context("test2", dummy_api.run_context, CONTEXT_TIMEOUT * 2)  # longer timeout
    manager.start()

    # Open both contexts
    await can_open_context(manager, ["test1", "test2"])
    
    # Both contexts should be open
    await is_open(manager, ["test1", "test2"])

    ctx1_id = await can_get_context_id(manager, "test1")
    ctx2_id = await can_get_context_id(manager, "test2")
    assert ctx1_id != ctx2_id, "Contexts should have different IDs"

    # Wait for the first context to expire, but not the second
    await asyncio.sleep(CONTEXT_TIMEOUT + 0.5)
    
    # First context should have expired, second should still be open
    await is_not_open(manager, ["test1"])
    await is_open(manager, ["test2"])

    # Reopen just the first context
    await can_open_context(manager, ["test1"])
    
    # Both should be open again
    await is_open(manager, ["test1", "test2"])

    # Close just the first context
    await can_close_context(manager, ["test1"])
    
    # First should be closed, second still open
    await is_not_open(manager, ["test1"])
    await is_open(manager, ["test2"])

    await can_shutdown(manager)


@pytest.mark.asyncio
async def test_context_manager_open_one_doesnt_affect_expiration(dummy_api: MockAPI):
    """Test that opening one context doesn't prevent others from expiring"""
    api = dummy_api

    manager = RuntimeContextManager()
    manager.add_context("test1", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.add_context("test2", dummy_api.run_context, CONTEXT_TIMEOUT)
    manager.start()

    # Open both contexts initially
    await can_open_context(manager, ["test1", "test2"])

    # Check that both contexts are open
    await is_open(manager, ["test1", "test2"])
    
    # Get the initial context IDs
    ctx1_id = await can_get_context_id(manager, "test1")
    ctx2_id = await can_get_context_id(manager, "test2")
    
    # Wait almost until timeout
    await asyncio.sleep(CONTEXT_TIMEOUT - 0.5)
    
    # Refresh only test1
    await can_open_context(manager, ["test1"])
    
    # Wait for test2 to expire
    await asyncio.sleep(1)
    
    # test1 should still be open with the same ID, test2 should have expired
    await is_open(manager, ["test1"])
    await is_not_open(manager, ["test2"])
    new_ctx1_id = await can_get_context_id(manager, "test1")
    assert new_ctx1_id == ctx1_id, "Context 1 should have the same ID"
    
    # Reopen test2
    await can_open_context(manager, ["test2"])
    
    # test2 should be open again but with a new ID
    await is_open(manager, ["test1", "test2"])
    new_ctx2_id = await can_get_context_id(manager, "test2")
    assert new_ctx2_id != ctx2_id, "Context 2 should have a new ID"

    await can_shutdown(manager)
