from abc import ABC, abstractmethod
from omegaconf import OmegaConf

from nexusvoice.ai.agents import AgentManager, Agent


class NexusAPI(ABC):
    """
    A class to interact with the Nexus API.
    """

    def __init__(self):
        pass

    @abstractmethod
    def agent_inference(self, agent_id, inputs):
        """
        Perform inference using the specified agent.
        :param agent_id: The ID of the agent to use for inference.
        :param inputs: The inputs to the agent.
        :return: The output of the agent.
        """
        pass
    
class NexusAPILocal(NexusAPI):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.agent_manager = AgentManager(config)

    def agent_inference(self, agent_id, inputs):
        agent = self.agent_manager.get_agent(agent_id)

        future_result = agent.process_request(inputs)

        # Wait for the result
        result = future_result.result()

        return result