import logfire
import pytest

from nexusvoice.core.config import load_config
from nexusvoice.client.RuntimeContextManager import RuntimeContextManager

import asyncio

import asyncio

class DummyContext:
    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

class SameTaskContext:
    def __init__(self):
        self._task = None
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self._task = asyncio.current_task()
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        exit_task = asyncio.current_task()
        if self._task != exit_task:
            raise Exception(f"__aexit__ called from a different task! Entered: {self._task}, Exited: {exit_task}")
        self.exited = True

class MockAPI:
    def __init__(self, context_provider):
        self.context_provider = context_provider
        self.context_created = False

    async def run_context(self):
        logfire.info("Providing context")
        self.context_created = True
        return self.context_provider()

@pytest.fixture
def dummy_context():
    return DummyContext()

@pytest.fixture
def same_task_context():
    return SameTaskContext()

@pytest.fixture
def dummy_api(dummy_context):
    return MockAPI(lambda: dummy_context)

@pytest.fixture
def same_task_api(same_task_context):
    return MockAPI(lambda: same_task_context)

async def is_not_open(manager: RuntimeContextManager):
    assert not manager._context_open, "Context should be closed"
    assert manager._context_opened_at is None, "Time should be unset when context is not open"
    assert manager.get_context() is None, "Context should be removed"

async def is_open(manager: RuntimeContextManager):
    assert manager.get_context() is not None
    assert manager._context_open, "Context should be open"
    assert manager._context_opened_at is not None, "Time should be set when context is opened"

async def can_open_context(manager: RuntimeContextManager):
    await manager.open()
    
    await is_open(manager)

async def can_close_context(manager: RuntimeContextManager):
    await manager.close()

    await is_not_open(manager)

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close(dummy_api):
    api = dummy_api
    manager = RuntimeContextManager(api, context_timeout=2)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    # Clean up
    manager.stop()
    try:
        task = manager.get_task()
        assert task is not None
        await task
        assert False, "Task was not cancelled"
    except asyncio.CancelledError as e:
        assert True, "Task was cancelled"

@pytest.mark.asyncio
async def test_runtime_context_manager_open_close_same_task(same_task_api):
    api = same_task_api

    manager = RuntimeContextManager(api, context_timeout=2)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)

    await can_close_context(manager)

    # Clean up
    manager.stop()
    try:
        task = manager.get_task()
        assert task is not None
        await task
        assert False, "Task was not cancelled"
    except asyncio.CancelledError as e:
        assert True, "Task was cancelled"

@pytest.mark.asyncio
async def test_context_manager_close_from_another_task(same_task_api):
    api = same_task_api

    manager = RuntimeContextManager(api, context_timeout=2)
    manager.start()

    await can_open_context(manager)

    # Close from a different task
    async def close_in_new_task():
        await manager.close()
    close_task = asyncio.create_task(close_in_new_task())
    await close_task  # Should NOT raise
    
    await is_not_open(manager)
    
    # Clean up
    manager.stop()
    try:
        task = manager.get_task()
        if task:
            await task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_context_manager_reopen_after_close(dummy_api):
    api = dummy_api

    manager = RuntimeContextManager(api, context_timeout=2)
    manager.start()

    await is_not_open(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)

    await can_open_context(manager)
    
    await can_close_context(manager)
    
