from nexusvoice.ai.pydantic_agent import HomeAutomationAgent, HomeAutomationResponse, HomeAutomationResponseStruct, NexusSupportDependencies
from nexusvoice.utils.debug import TimeThis
from pydantic_ai import RunContext
from nexusvoice.core.config import load_config
from pydantic_ai.providers.openai import OpenAIProvider

import logging
import pytest

@pytest.fixture(autouse=True, scope="session")
def enable_debug_logging():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

@pytest.fixture
def home_control_tool():
    called = {"value": False, "args": None}
    def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> str:
        called["value"] = True
        called["args"] = {"intent": intent, "device": device, "room": room}
        return f"{intent} {device} in {room}"
    return home_control, called

def test_home_automation_basic():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = HomeAutomationAgent(support_deps)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    summary_message = HomeAutomationResponseStruct.extract_message(response)
    assert "living room" in summary_message
    assert "on" in summary_message


def test_home_automation_tool(home_control_tool):
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = HomeAutomationAgent(support_deps)
    
    home_control, called = home_control_tool
    agent.register_tool(home_control)
    
    # Test home automation request
    with TimeThis("test_home_automation_tool"):
        result = agent.run_sync("Turn on the living room lights")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    summary_message = HomeAutomationResponseStruct.extract_message(response)
    assert "living room" in summary_message
    assert "on" in summary_message
    assert called["value"], "Home automation tool was not executed"
    assert called['args']['room'], "Room not specified"
    assert called['args']['intent'], "Intent not specified"
    assert called['args']['device'], "Device not specified"
    assert called['args']['room'] == "living room"
    assert "on" in called['args']['intent']
    assert "light" in called['args']['device']

def test_home_automation_custom_provider(home_control_tool):
    config = load_config()
    config.set("agents.home_automation.model", "llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b")

    provider = OpenAIProvider(
        api_key=config.get("openai.api_key", ""),
        base_url="http://localhost:1234/v1/"
    )
    
    support_deps = NexusSupportDependencies(config=config)
    agent = HomeAutomationAgent(support_deps, provider=provider)
    
    home_control, called = home_control_tool
    agent.register_tool(home_control)
    
    # Test home automation request
    with TimeThis("test_home_automation_custom_provider"):
        result = agent.run_sync("Turn on the living room lights")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    summary_message = HomeAutomationResponseStruct.extract_message(response)
    
    assert called["value"], "Home automation tool was not executed"
    assert called['args']['room'], "Room not specified"
    assert called['args']['intent'], "Intent not specified"
    assert called['args']['device'], "Device not specified"
    assert called['args']['room'] == "living room"
    assert "on" in called['args']['intent']
    assert "light" in called['args']['device']

    assert "living room" in summary_message
    assert "on" in summary_message