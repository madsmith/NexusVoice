from nexusvoice.ai.pydantic_agent import PydanticAgent
from nexusvoice.core.api.base import ModelResponse
from nexusvoice.core.config import load_config
from pydantic_ai.messages import TextPart
import pytest

def test_pydantic_agent_must_be_started():
    config = load_config()
    agent = PydanticAgent(config, "test_client_id")
    with pytest.raises(AssertionError):
        agent.classifier_agent
    with pytest.raises(AssertionError):
        agent.home_agent
    with pytest.raises(AssertionError):
        agent.conversational_agent
    agent.start()
    assert agent.classifier_agent is not None
    assert agent.home_agent is not None
    assert agent.conversational_agent is not None


def test_pydantic_agent_process_request_home_automation():
        config = load_config()
        agent = PydanticAgent(config, "test_client_id")
        agent.start()
        
        response = agent.process_request("Turn on the living room lights")

        assert isinstance(response, ModelResponse)
        assert isinstance(response.parts[0], TextPart)
        assert "living room" in response.parts[0].content
        assert "on" in response.parts[0].content


def test_pydantic_agent_process_request_conversational():
    config = load_config()
    agent = PydanticAgent(config, "test_client_id")
    agent.start()

    response = agent.process_request("What is the capital of Ohio?")

    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], TextPart)
    assert "Ohio" in response.parts[0].content
    assert "capital" in response.parts[0].content
    assert "Columbus" in response.parts[0].content
    