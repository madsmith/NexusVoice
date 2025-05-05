from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.ai.types import RequestType, NexusSupportDependencies
from pydantic_ai.agent import AgentRunResult
from nexusvoice.core.config import load_config

def test_local_classifier():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = LocalClassifierAgentFactory.create(support_deps)
    
    # Test home automation request
    result = agent.run_sync("Turn on the living room lights", deps=support_deps)

    assert isinstance(result, AgentRunResult)
    assert isinstance(result.data, RequestType)

    classification = result.data
    assert classification.type == "home_automation"
    assert classification.confidence > 0.7
        
    # Test conversation request
    result = agent.run_sync("What's the weather like today?", deps=support_deps)

    assert isinstance(result, AgentRunResult)
    assert isinstance(result.data, RequestType)

    classification = result.data
    assert classification.type == "conversation"
    assert classification.confidence > 0.5