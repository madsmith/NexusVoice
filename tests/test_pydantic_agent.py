from pydantic_ai import RunContext
import pytest
import asyncio
from nexusvoice.ai.pydantic_agent import ConversationalAgent, HomeAutomationAgent, HomeAutomationResponse, LocalClassifierAgent, NexusSupportDependencies, RequestType
from nexusvoice.core.config import load_config

def test_local_classifier():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalClassifierAgent(support_deps)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights")

    classification = result.data
    assert classification.type == "home_automation"
    assert classification.confidence > 0.7
    
    # Test conversation request
    result = agent.run_sync("What's the weather like today?")
    
    classification = result.data
    assert classification.type == "conversation"
    assert classification.confidence > 0.5


def test_home_automation_basic():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = HomeAutomationAgent(support_deps)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    assert "living room" in response.text
    assert "on" in response.text


def test_conversation():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = ConversationalAgent(support_deps)
    
    # Test conversation request
    result = agent.run_sync("What's the weather like today?")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    assert "weather" in response.text
    assert "today" in response.text

# Fix for pydantic_ai using get_event_loop() in synchronous functions 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
