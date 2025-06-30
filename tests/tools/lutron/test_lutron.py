import asyncio
import logging
import os
import sys
import pytest
import pytest_asyncio

from nexusvoice.bootstrap import bootstrap # Side effects on import
from nexusvoice.core.config import load_config

config = load_config()

from nexusvoice.tools.lutron.lutron import LutronHomeworksClient

@pytest_asyncio.fixture()
async def lutron_client():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stdout,
    )

    host = config.get('tools.lutron.host')
    port = int(config.get('tools.lutron.port'))
    username = config.get('tools.lutron.username')
    password = config.get('tools.lutron.password')
    keepalive_interval = 10

    client = LutronHomeworksClient(
        host=host,
        username=username,
        password=password,
        port=port,
        keepalive_interval=keepalive_interval,
    )

    yield client
    
    # Ensure the client is closed after tests
    await client.close()

@pytest.mark.asyncio
async def test_connect_and_command(lutron_client):
    await lutron_client.connect()
    assert lutron_client.connected, "Should connect successfully to Lutron server"

    system_response_holder = {}
    lutron_client.subscribe('system_response', lambda data: system_response_holder.update(data))

    await lutron_client.send_command('?SYSTEM,1')
    await asyncio.sleep(0.5)  # Wait for any responses/keepalive
    
    print(system_response_holder)

@pytest.mark.asyncio
async def test_monitor_events_for_duration(lutron_client):
    await lutron_client.connect()
    assert lutron_client.connected, "Should connect successfully to Lutron server"
    
    await asyncio.sleep(30)  # Wait for any responses/keepalive

    await lutron_client.close()

