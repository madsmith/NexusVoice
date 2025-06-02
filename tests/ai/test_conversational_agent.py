from nexusvoice.ai.ConversationalAgent import ConversationalAgentFactory

from nexusvoice.ai.types import ConversationResponse, NexusSupportDependencies
from nexusvoice.core.config import load_config

def test_conversational_agent_basic():
    config = load_config()
    support_deps = NexusSupportDependencies(config=config)
    agent = ConversationalAgentFactory.create(support_deps)
    
    # Test conversation request
    result = agent.run_sync("What's the capital of the Netherlands?", deps=support_deps)

    response = result.output
    assert isinstance(response, ConversationResponse)
    assert "Amsterdam" in response.text
