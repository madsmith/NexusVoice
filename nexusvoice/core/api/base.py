from abc import ABC, abstractmethod
from nexusvoice.core.config import NexusConfig
from pydantic_ai.messages import ModelResponse as PydanticModelResponse
from contextlib import AbstractAsyncContextManager, AsyncExitStack

class ModelResponse(PydanticModelResponse):
    @property
    def text(self) -> str:
        return "".join(getattr(part, "content", "") for part in self.parts)


from typing import TypeVar, Generic

T = TypeVar("T", bound="NexusAPIContext")

class NexusAPIContext(AbstractAsyncContextManager, Generic[T], ABC):
    """
    Abstract base for all API session contexts.
    Concrete subclasses should define fields/resources relevant to their implementation.
    Implements a generic async context manager protocol so that __aenter__ returns the concrete type.
    """
    @abstractmethod
    async def __aenter__(self) -> T:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# For backward compatibility, alias NexusContextManager to NexusAPIContext
NexusContextManager = NexusAPIContext

class NexusAPI(ABC):
    """A class to interact with the Nexus API."""
    
    def __init__(self, config: NexusConfig):
        self._config = config

    @property
    def config(self) -> NexusConfig:
        return self._config

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def run_context(self) -> NexusAPIContext:
        """
        Returns an async context manager for a Nexus API session.
        Should yield a NexusAPIContext instance, which holds session state/resources.
        """
        pass
    
    @abstractmethod
    async def prompt_agent(self, agent_id: str, prompt: str) -> str:
        """
        Prompt the specified agent with the given prompt.

        :param agent_id: The ID of the agent to use for inference.
        :param prompt: The prompt to send to the agent.
        :return: The output of the agent.
        """
        pass