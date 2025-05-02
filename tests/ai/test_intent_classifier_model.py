import pytest
from nexusvoice.ai.LocalModel import IntentClassifierModel, LocalClassifierAgentFactory
from nexusvoice.ai.types import NexusSupportDependencies, RequestType
from pydantic_ai.agent import Agent, AgentRunResult
from nexusvoice.core.config import load_config

# This test directly instantiates an Agent with a custom IntentClassifierModel

# Shared setup for tests
@pytest.fixture
def setup_agent():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    model = IntentClassifierModel(config)
    agent = Agent[NexusSupportDependencies, RequestType](
        model=model,
        deps_type=NexusSupportDependencies,
        result_type=RequestType,
    )
    return agent, support_deps

def test_intent_classifier_home_automation(setup_agent):
    agent, support_deps = setup_agent
    result = agent.run_sync("Turn on the living room lights", deps=support_deps)
    assert isinstance(result, AgentRunResult)
    assert isinstance(result.data, RequestType)
    classification = result.data
    assert classification.type == "home_automation"
    assert classification.confidence > 0.7

def test_intent_classifier_conversation(setup_agent):
    agent, support_deps = setup_agent
    result = agent.run_sync("What's the weather like today?", deps=support_deps)
    assert isinstance(result, AgentRunResult)
    assert isinstance(result.data, RequestType)
    classification = result.data
    assert classification.type == "conversation"
    assert classification.confidence > 0.5

def test_local_classifier_agent_factory():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalClassifierAgentFactory.create(support_deps)
    assert isinstance(agent, Agent)
    # Optionally, run a simple classification to check integration
    result = agent.run_sync("Turn on the kitchen lights", deps=support_deps)
    assert isinstance(result, AgentRunResult)
    assert isinstance(result.data, RequestType)
    classification = result.data
    assert classification.type == "home_automation"
    assert classification.confidence > 0.7
