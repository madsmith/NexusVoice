import pytest
from nexusvoice.core.api.online import NexusAPIOnline
from nexusvoice.core.config import load_config

@pytest.mark.asyncio
async def test_nexus_api_initialization():
    config = load_config()
    api = NexusAPIOnline(config)
    await api.initialize()
    # Ensure agents are initialized
    assert api.classifier_agent is not None
    assert api.home_agent is not None
    assert api.conversational_agent is not None

@pytest.mark.asyncio
async def test_nexus_api_home_automation():
    config = load_config()
    api = NexusAPIOnline(config)
    await api.initialize()
    # Test a home automation prompt
    response = await api.prompt_agent("test_client_id", "Turn on the living room lights")
    assert isinstance(response, str)
    assert "living room" in response.lower()
    assert "on" in response.lower()

@pytest.mark.asyncio
async def test_nexus_api_conversation():
    config = load_config()
    api = NexusAPIOnline(config)
    await api.initialize()
    # Test a general conversation prompt
    response = await api.prompt_agent("test_client_id", "What's the weather like today?")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_nexus_mcp_servers():
    config = load_config()
    api = NexusAPIOnline(config)
    await api.initialize()
    # assert api._mcp_servers and len(api._mcp_servers) > 0, "No MCP servers initialized"
    # assert api._deps.servers and len(api._deps.servers) > 0, "No MCP servers initialized to deps"

    # response = await api.prompt_agent("test_client_id", "What is 3x + 45 where x is 13?")
    response = await api.prompt_agent("test_client_id", "How many days between 2000-01-01 and 2025-03-18?")
    assert isinstance(response, str)
    assert "9,208" in response or "9208" in response, "Failed to get days between dates"
    

