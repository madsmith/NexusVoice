from nexusvoice.ai.pydantic_agent import ConversationalAgent, ConversationResponse, NexusSupportDependencies
from nexusvoice.core.config import load_config

def test_conversational_agent_basic():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = ConversationalAgent(support_deps)
    
    # Test conversation request
    result = agent.run_sync("What's the capital of the Netherlands?")

    response = result.data
    assert isinstance(response, ConversationResponse)
    assert "Amsterdam" in response.text
