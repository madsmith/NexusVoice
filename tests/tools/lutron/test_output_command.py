import logging
import sys
import asyncio
import pytest
import pytest_asyncio

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from nexusvoice.tools.lutron.lutron import LutronHomeworksClient
from nexusvoice.tools.lutron.commands.output import OutputCommand
from nexusvoice.core.config import load_config

config = load_config()

TEST_IID = 179
TEST_SHADE_IID = 356

@pytest_asyncio.fixture()
async def lutron_client():
    """Setup fixture for Lutron client, similar to test_lutron.py"""
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

    await client.connect()
    
    yield client
    
    # Ensure the client is closed after tests
    await client.close()


# Manual test for executing an OutputCommand directly (can be run in VSCode directly)
@pytest.mark.asyncio
async def test_manual_output_command(lutron_client):
    """Test executing an OutputCommand manually to get zone level."""
    # Create the command for IID
    cmd = OutputCommand.get_zone_level(TEST_IID)
    
    # Print command details for debugging
    print(f"Command: {cmd}")
    print(f"Command formatted: {cmd.formatted_command}")

    print("Response Map: ", cmd.schema.response_index_map)

    assert cmd.formatted_command == f"?OUTPUT,{TEST_IID},1"
    
    # Execute the command against the Lutron client
    print("Executing command...")
    await asyncio.sleep(3)
    result = await cmd.execute(lutron_client)
    
    # Print the result
    print(f"Command result: {result}")
    print(f"Result type: {type(result)}")
    
    # Return for further inspection
    return result


# Test for getting zone level
@pytest.mark.asyncio
async def test_get_zone_level(lutron_client):
    """Test OutputCommand.get_zone_level() to get the current level of a zone (IID)."""
    # Create the output zone level command for IID
    cmd = OutputCommand.get_zone_level(TEST_IID)
    
    # Execute the command against a real Lutron system
    result = await cmd.execute(lutron_client)
    
    # The result should be a float between 0 and 100
    assert result is not None
    print(f"Zone {TEST_IID} level: {result}")
    
    # Verify it has the expected components
    assert isinstance(result, float)
    assert 0 <= result <= 100


# Check for the format of a set_level command without executing it
@pytest.mark.asyncio
async def test_set_zone_level_format():
    """Test OutputCommand.set_zone_level() to set the level of a zone (IID)."""
    # Create the output zone level command for IID
    cmd = OutputCommand.set_zone_level(TEST_IID, 50.0)
    
    assert f"#OUTPUT,{TEST_IID},1,50" in cmd.formatted_command

# Test for setting zone level
@pytest.mark.asyncio
async def test_set_and_get_zone_level(lutron_client):
    """Test setting and then getting a zone level to verify the change."""
    # Create the commands for IID
    
    # First get current level
    get_cmd = OutputCommand.get_zone_level(TEST_IID)
    original_level = await get_cmd.execute(lutron_client)
    print(f"Original level of zone {TEST_IID}: {original_level}")
    
    # Set to a different level (50% if original wasn't 50%, otherwise 25%)
    new_level = 75.0 if original_level != 75.0 else 25.0
    set_cmd = OutputCommand.set_zone_level(TEST_IID, new_level)
    await set_cmd.execute(lutron_client)
    print(f"Set zone {TEST_IID} to level: {new_level}")
    
    # Wait a moment for command to take effect
    await asyncio.sleep(2)
    
    # Get level again to verify
    get_cmd = OutputCommand.get_zone_level(TEST_IID)
    updated_level = await get_cmd.execute(lutron_client)
    print(f"Updated level of zone {TEST_IID}: {updated_level}")
    
    # Verify level was set (allow small tolerance for floating point)
    assert abs(updated_level - new_level) < 0.1
    
    # Set back to original level
    set_cmd = OutputCommand.set_zone_level(TEST_IID, original_level)
    await set_cmd.execute(lutron_client)
    print(f"Restored zone {TEST_IID} to original level: {original_level}")

@pytest.mark.asyncio
async def test_output_raise_lower(lutron_client):
    """Test OutputCommand.start_raise() to start raising a zone."""
    cmd = OutputCommand.start_raise(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None

    await asyncio.sleep(5)
    
    cmd = OutputCommand.stop_raise_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None
    
    await asyncio.sleep(.2)

    cmd = OutputCommand.start_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None
    
    await asyncio.sleep(5)
    
    cmd = OutputCommand.stop_raise_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None

    await asyncio.sleep(.2)
    
    cmd = OutputCommand.start_raise(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None

    await asyncio.sleep(.2)
    
    cmd = OutputCommand.stop_raise_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None
    
    await asyncio.sleep(.2)

    cmd = OutputCommand.start_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None
    
    await asyncio.sleep(.2)
    
    cmd = OutputCommand.stop_raise_lower(TEST_SHADE_IID)
    result = await cmd.execute(lutron_client)
    assert result is None

    await asyncio.sleep(.2)
    
    
    
    
    
    
    