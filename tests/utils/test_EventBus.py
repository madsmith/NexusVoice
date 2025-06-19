import pytest
import asyncio
from nexusvoice.utils.eventbus import EventBus

@pytest.mark.asyncio
async def test_basic_emit_and_subscribe():
    bus = EventBus()
    received = []

    async def callback(data):
        received.append(data)

    token = bus.subscribe('test_event', callback)
    await bus.emit('test_event', 'hello')
    # Let the event loop run to process the callback
    await asyncio.sleep(0.01)
    assert received == ['hello']

    # Test unsubscribe
    assert bus.unsubscribe(token) is True
    await bus.emit('test_event', 'world')
    await asyncio.sleep(0.01)
    assert received == ['hello']  # Should not change
    # Unsubscribing again should return False
    assert bus.unsubscribe(token) is False

@pytest.mark.asyncio
async def test_multiple_callbacks():
    bus = EventBus()
    received1 = []
    received2 = []

    async def cb1(data):
        received1.append(data)
    async def cb2(data):
        received2.append(data)

    token1 = bus.subscribe('evt', cb1)
    token2 = bus.subscribe('evt', cb2)
    await bus.emit('evt', 42)
    await asyncio.sleep(0.01)
    assert received1 == [42]
    assert received2 == [42]

    # Unsubscribe one and check only the other is called
    assert bus.unsubscribe(token1) is True
    await bus.emit('evt', 99)
    await asyncio.sleep(0.01)
    assert received1 == [42]  # No change
    assert received2 == [42, 99]

    # Unsubscribe the second
    assert bus.unsubscribe(token2) is True
    await bus.emit('evt', 123)
    await asyncio.sleep(0.01)
    assert received2 == [42, 99]  # No change

@pytest.mark.asyncio
async def test_once():
    bus = EventBus()
    received = []

    async def cb(data):
        received.append(data)

    token = bus.once('evt', cb)
    await bus.emit('evt', 1)
    await asyncio.sleep(0.01)
    assert received == [1]
    await bus.emit('evt', 2)
    await asyncio.sleep(0.01)
    assert received == [1]  # Should not be called again
    # Unsubscribing after it has already been called should return False
    assert bus.unsubscribe(token) is False

@pytest.mark.asyncio
async def test_no_callbacks():
    bus = EventBus()
    # Should not raise
    await bus.emit('no_listeners', 'data')

@pytest.mark.asyncio
async def test_sync_and_async_callbacks():
    """Test that both synchronous and asynchronous callbacks work properly."""
    bus = EventBus()
    sync_received = []
    async_received = []
    sync_delayed_received = []
    
    # Synchronous callback
    def sync_callback(data):
        sync_received.append(data)
    
    # Asynchronous callback
    async def async_callback(data):
        # Simulate some async processing
        await asyncio.sleep(0.01)
        async_received.append(data)
    
    # Synchronous callback that returns a coroutine
    def sync_callback_returns_coroutine(data):
        async def delayed_processing():
            await asyncio.sleep(0.02)
            sync_delayed_received.append(data)
        return delayed_processing()
    
    # Subscribe all callbacks to the same event
    token1 = bus.subscribe('mixed_event', sync_callback)
    token2 = bus.subscribe('mixed_event', async_callback)
    token3 = bus.subscribe('mixed_event', sync_callback_returns_coroutine)
    
    # Emit event
    await bus.emit('mixed_event', 'test_data')
    
    # Let event loop process all callbacks
    await asyncio.sleep(0.05)
    
    # Verify all callbacks were executed
    assert sync_received == ['test_data']
    assert async_received == ['test_data']
    assert sync_delayed_received == ['test_data']
    
    # Test unsubscribing
    assert bus.unsubscribe(token1) is True
    assert bus.unsubscribe(token2) is True
    assert bus.unsubscribe(token3) is True
    
    # Clear lists and emit again
    sync_received.clear()
    async_received.clear()
    sync_delayed_received.clear()
    
    await bus.emit('mixed_event', 'new_data')
    await asyncio.sleep(0.05)
    
    # Verify no callbacks were executed
    assert sync_received == []
    assert async_received == []
    assert sync_delayed_received == []
