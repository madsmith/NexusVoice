from abc import ABC, abstractmethod
from pydantic_ai.messages import ModelResponse as PydanticModelResponse

class ModelResponse(PydanticModelResponse):
    @property
    def text(self) -> str:
        return "".join(getattr(part, "content", "") for part in self.parts)

class NexusAPI(ABC):
    """A class to interact with the Nexus API."""
    
    def __init__(self):
        pass

    @abstractmethod
    def agent_inference(self, agent_id: str, inputs) -> str:
        """
        Perform inference using the specified agent.
        :param agent_id: The ID of the agent to use for inference.
        :param inputs: The inputs to the agent.
        :return: The output of the agent.
        """
        pass

    @abstractmethod
    def mcp_agent_inference(self, agent_id: str, inputs) -> ModelResponse:
        """
        Perform inference using the specified agent with the MCP Protocol.
        :param agent_id: The ID of the agent to use for inference.
        :param inputs: The input message in MCP format.
        :return: The output message in MCP format.
        """
        pass