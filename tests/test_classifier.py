import asyncio
from pydantic_ai.agent import AgentRunResult
import pytest
from nexusvoice.ai.pydantic_agent import HomeAutomationAction, HomeAutomationAgent, LocalClassifierAgent, FastClassifierAgent, RequestType, process_request

from nexusvoice.core.config import load_config
from nexusvoice.utils.debug import TimeThis

def test_local_classifier():
    with TimeThis("LocalClassifierAgent"):
        config = load_config()
        agent = LocalClassifierAgent(config)
    
        # Test home automation request
        result = agent.run_sync("Turn on the living room lights")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, RequestType)

        classification = result.data
        assert classification.type == "home_automation"
        assert classification.confidence > 0.7
        
        # Test conversation request
        result = agent.run_sync("What's the weather like today?")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, RequestType)

        classification = result.data
        assert classification.type == "conversation"
        assert classification.confidence > 0.5

def test_fast_classifier():
    with TimeThis("FastClassifierAgent"):
        config = load_config()
        agent = FastClassifierAgent(config)
    
        # Test home automation request
        result = agent.run_sync("Turn on the living room lights")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, RequestType)

        classification = result.data
        assert classification.type == "home_automation"
        assert classification.confidence > 0.7
        
        # Test conversation request
        result = agent.run_sync("What's the weather like today?")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, RequestType)

        classification = result.data
        assert classification.type == "conversation"
        assert classification.confidence > 0.5

def test_home_automation():
    with TimeThis("HomeAutomationAgent"):
        config = load_config()
        agent = HomeAutomationAgent(config)
        
        # Test home automation request
        result = agent.run_sync("Turn on the living room lights")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, HomeAutomationAction)

        action = result.data
        assert action.intent == "turn_on"
        assert action.device == "light"
        assert action.room == "living room"

def test_process_request():
    with TimeThis("process_request"):
        config = load_config()
        return
        response = process_request(config, "Turn on the living room lights")
        print(response)

# Fix for pydantic_ai using get_event_loop() in synchronous functions 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
