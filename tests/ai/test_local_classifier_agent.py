from nexusvoice.ai.pydantic_agent import LocalClassifierAgent, RequestType, NexusSupportDependencies
from pydantic_ai.agent import AgentRunResult
from nexusvoice.core.config import load_config

def test_local_classifier():
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