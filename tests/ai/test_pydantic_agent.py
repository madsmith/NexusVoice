from nexusvoice.ai.pydantic_agent import PydanticAgent
from nexusvoice.core.config import load_config

def test_pydantic_agent_process_request():
        config = load_config()
        agent = PydanticAgent(config, "test_client_id")
        agent.start()
        response = agent.process_request("Turn on the living room lights")
        assert False, "Not Implemented"