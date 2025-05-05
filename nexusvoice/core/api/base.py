from abc import ABC, abstractmethod
from nexusvoice.core.config import NexusConfig
from pydantic_ai.messages import ModelResponse as PydanticModelResponse

class ModelResponse(PydanticModelResponse):
    @property
    def text(self) -> str:
        return "".join(getattr(part, "content", "") for part in self.parts)

class NexusAPI(ABC):
    """A class to interact with the Nexus API."""
    
    def __init__(self, config: NexusConfig):
        self._config = config

    @property
    def config(self) -> NexusConfig:
        return self._config

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def prompt_agent(self, agent_id: str, prompt: str) -> str:
        """
        Prompt the specified agent with the given prompt.

        :param agent_id: The ID of the agent to use for inference.
        :param prompt: The prompt to send to the agent.
        :return: The output of the agent.
        """
        pass