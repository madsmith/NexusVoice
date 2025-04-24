import asyncio
from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
import pytest
from nexusvoice.ai.pydantic_agent import (
    HomeAutomationAgent,
    HomeAutomationResponse,
    LocalClassifierAgent,
    FastClassifierAgent,
    NexusSupportDependencies,
    PydanticAgent,
    RequestType
)

from nexusvoice.core.config import load_config
from nexusvoice.utils.debug import TimeThis

def test_local_classifier():
    with TimeThis("LocalClassifierAgent"):
        config = load_config()
        support_deps = NexusSupportDependencies(config=config)
        agent = LocalClassifierAgent(support_deps)
    
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
        support_deps = NexusSupportDependencies(config=config)
        agent = FastClassifierAgent(support_deps)
    
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
    def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> str:
        """
        Turn on, off, raise, or lower a home automation device.
        
        Args:
            ctx: The run context
            intent: The action to perform (e.g., turn_on, turn_off, raise, lower)
            device: The device to control (e.g., light, fan, shade)
            room: The room where the device is located
        """
        print(f"Home Automation: {intent} {device} in {room}")
        return "Done"
        
    with TimeThis("HomeAutomationAgent"):
        config = load_config()
        support_deps = NexusSupportDependencies(config=config)
        agent = HomeAutomationAgent(support_deps)
        agent.register_tool(home_control)
        
        # Test home automation request
        result = agent.run_sync("Turn on the living room lights")

        assert isinstance(result, AgentRunResult)
        assert isinstance(result.data, HomeAutomationResponse)


def test_pydantic_agent_process_request():
    with TimeThis("process_request"):
        config = load_config()
        agent = PydanticAgent(config, "test_client_id")
        agent.start()
        response = agent.process_request("Turn on the living room lights")
        assert False, "Not Implemented"

# Fix for pydantic_ai using get_event_loop() in synchronous functions 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
