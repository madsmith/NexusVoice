import pytest
import asyncio
from nexusvoice.ai.pydantic_agent import LocalClassifierAgent, RequestType
from nexusvoice.core.config import load_config

def test_local_classifier():
    config = load_config()
    agent = LocalClassifierAgent(config)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights")

    classification = RequestType.model_validate_json(result.data)
    assert classification.type == "home_automation"
    assert classification.confidence > 0.7
    
    # Test conversation request
    result = agent.run_sync("What's the weather like today?")
    classification = RequestType.model_validate_json(result.data)
    assert classification.type == "conversation"
    assert classification.confidence > 0.5

# Fix for pydantic_ai using get_event_loop() in synchronous functions 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
