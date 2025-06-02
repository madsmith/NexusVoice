from pydantic_ai import RunContext
from pydantic_graph import End
import pytest
from nexusvoice.ai.LocalHomeAutomationAgent import HomeAutomationModel, LocalHomeAutomationAgentFactory
from nexusvoice.ai.types import HomeAutomationResponseStruct, NexusSupportDependencies, HomeAutomationResponse
from pydantic_ai.agent import Agent, AgentRunResult
from nexusvoice.core.config import load_config


@pytest.fixture
def home_control_tool():
    called = {"value": False, "args": None}
    def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> str:
        called["value"] = True
        called["args"] = {"intent": intent, "device": device, "room": room}
        # print("=" * 80)
        # print("=" * 80)
        # print("Home control called with args:", called["args"])
        # print("=" * 80)
        # print("=" * 80)
        return f"{intent} {device} in {room}"
    return home_control, called

@pytest.mark.asyncio
async def test_local_home_automation_agent_factory(home_control_tool):
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalHomeAutomationAgentFactory.create(support_deps)
    assert isinstance(agent, Agent)
    
    home_control, called = home_control_tool
    agent.tool(home_control)

    # Optionally, run a simple classification to check integration
    result = await agent.run("Turn on the kitchen lights", deps=support_deps)
    
    print(result)
    assert isinstance(result, AgentRunResult)
    assert isinstance(result.output, HomeAutomationResponse)
    assert called['value'], "Home control tool was not called"
    assert "kitchen" in called['args']['room']
    assert "on" in called['args']['intent']
    assert "light" in called['args']['device']
    summary_message = HomeAutomationResponseStruct.extract_message(result.output)
    assert "kitchen" in summary_message.lower()
    assert "on" in summary_message.lower()
    

@pytest.mark.asyncio
async def test_local_home_automation_agent_iter_valid(home_control_tool):
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalHomeAutomationAgentFactory.create(support_deps)
    assert isinstance(agent, Agent)
    
    home_control, called = home_control_tool
    agent.tool(home_control)

    # Optionally, run a simple classification to check integration
    # result = agent.run_sync("Turn on the kitchen lights", deps=support_deps)
    async with agent.iter('Turn on the kitchen lights', deps=support_deps) as agent_run:
        next_node = agent_run.next_node  # start with the first node
        nodes = [next_node]
        while not isinstance(next_node, End):
            next_node = await agent_run.next(next_node)
            nodes.append(next_node)
        # Once `next_node` is an End, we've finished:
        print("\n== Summary of Agent Run ==")
        for node in nodes:
            print(node)
            print()
        print("== End of Agent Run ==")
    # print(result)
    # assert isinstance(result, AgentRunResult)
    # assert isinstance(result.output, HomeAutomationResponse)
    # assert called['value'], "Home control tool was not called"


@pytest.mark.asyncio
async def test_local_home_automation_agent_factory_iter_invalid(home_control_tool):
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalHomeAutomationAgentFactory.create(support_deps)
    assert isinstance(agent, Agent)
    
    home_control, called = home_control_tool
    agent.tool(home_control)

    # Optionally, run a simple classification to check integration
    # result = agent.run_sync("Turn on the kitchen lights", deps=support_deps)
    async with agent.iter('simon says hello', deps=support_deps) as agent_run:
        next_node = agent_run.next_node  # start with the first node
        nodes = [next_node]
        while not isinstance(next_node, End):
            next_node = await agent_run.next(next_node)
            nodes.append(next_node)
        # Once `next_node` is an End, we've finished:
        print("\n== Summary of Agent Run ==")
        for node in nodes:
            print(node)
            print()
        print("== End of Agent Run ==")
    # print(result)
    # assert isinstance(result, AgentRunResult)
    # assert isinstance(result.output, HomeAutomationResponse)
    # assert called['value'], "Home control tool was not called"

def test_local_home_automation_agent_factory_invalid_intent(home_control_tool):
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalHomeAutomationAgentFactory.create(support_deps)
    assert isinstance(agent, Agent)
    
    home_control, called = home_control_tool
    agent.tool(home_control)

    # Optionally, run a simple classification to check integration
    result = agent.run_sync("What's the weather like today?", deps=support_deps)

    print(result)
    assert isinstance(result, AgentRunResult)
    assert isinstance(result.output, HomeAutomationResponse)
    assert called['value'], "Home control tool was not called"

