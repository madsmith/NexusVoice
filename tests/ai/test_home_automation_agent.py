from nexusvoice.ai.pydantic_agent import HomeAutomationAgent, HomeAutomationResponse, NexusSupportDependencies
from pydantic_ai import RunContext
from nexusvoice.core.config import load_config

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


def test_home_automation_tool():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = HomeAutomationAgent(support_deps)
    
    tool_executed = False
    def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> str:
        nonlocal tool_executed
        tool_executed = True
        return f"{intent} {device} in {room}"

    agent.register_tool(home_control)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights")

    response = result.data
    assert isinstance(response, HomeAutomationResponse)
    assert "living room" in response.text
    assert "on" in response.text
    assert tool_executed, "Home automation tool was not executed"
