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
