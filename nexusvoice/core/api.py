from abc import ABC, abstractmethod
from omegaconf import OmegaConf

import requests
from typing import List

from nexusvoice.ai.agents import AgentManager, Agent
from nexusvoice.protocol.mcp import UserMessage, ModelMessage, ToolResult
from nexusvoice.utils.debug import TimeThis
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

tool_registry = {}

def register_tool(tool_name: str):
    """
    Decorator to register a tool function.
    :param tool_name: The name of the tool.
    """
    def decorator(func):
        tool_registry[tool_name] = func
        return func
    return decorator

@register_tool("get_weather")

def get_weather_tool(input: dict):
    city = input.get("city", "New York")
    config = OmegaConf.load("config.yml")
    key = config.tools.weather.get("api_key", None)

    if not key:
        raise ValueError("API key for weather tool not found in config.")

    resp = requests.get("https://api.weatherapi.com/v1/current.json", params={
        "key": key,
        "q": city
    })
    data = resp.json()

    return {
        "city": data["location"]["name"],
        "temp_f": data["current"]["temp_f"],
        "condition": data["current"]["condition"]["text"]
    }

class NexusAPI(ABC):
    """
    A class to interact with the Nexus API.
    """

    def __init__(self):
        pass

    @abstractmethod
    def agent_inference(self, agent_id: str, inputs):
        """
        Perform inference using the specified agent.
        :param agent_id: The ID of the agent to use for inference.
        :param inputs: The inputs to the agent.
        :return: The output of the agent.
        """
        pass

    @abstractmethod
    def mcp_agent_inference(self, agent_id: str, mcp_input: UserMessage) -> ModelMessage:
        """
        Perform inference using the specified agent with the MCP Protocol.
        :param agent_id: The ID of the agent to use for inference.
        :param mcp_input: The input message in MCP format.
        :return: The output message in MCP format.
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
    
    def mcp_agent_tool_response(self, agent_id: str, tool_results: List[ToolResult]) -> ModelMessage:
        agent = self.agent_manager.get_agent(agent_id)

        future_result = agent.process_tool_response(tool_results)

        # Wait for the result
        result = future_result.result()

        return result
    

    def mcp_agent_inference(self, agent_id: str, mcp_input: UserMessage) -> ModelMessage:
        result_text = self.agent_inference(agent_id, mcp_input.text)

        logger.debug(f"Agent Inference Result: \n========\n{result_text}\n========")

        # Try to parse as ToolCall
        try:
            parsed = ModelMessage.model_validate_json(result_text)
        except Exception:
            return ModelMessage(text=result_text)

        if parsed.tool_calls:
            tool_results = []
            for call in parsed.tool_calls:
                with TimeThis(f"Tool Call: {call.tool_name}"):
                    tool_fn = tool_registry.get(call.tool_name)
                    if tool_fn:
                        output = tool_fn(call.input)
                        logger.debug(f"Tool Call Result: \n========\n{output}\n========")
                        tool_results.append(ToolResult(id=call.id, output=output))
                    else:
                        tool_results.append(ToolResult(id=call.id, output={"error": "Tool not found"}))

            return self.mcp_agent_tool_response(agent_id, tool_results)

        else:
            return parsed