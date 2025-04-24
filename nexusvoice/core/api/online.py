from dataclasses import dataclass
from typing import List
from nexusvoice.core.api import NexusAPI
from nexusvoice.core.config import NexusConfig
from pydantic_ai import RunContext
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from nexusvoice.core.api.base import ModelResponse
from nexusvoice.utils.debug import TimeThis
from nexusvoice.utils.logging import get_logger
from nexusvoice.protocol import mcp

logger = get_logger(__name__)

from nexusvoice.ai.pydantic_agent import NexusSupportDependencies, PydanticAgent
from nexusvoice.core.api.tool_registry import tool_registry

from nexusvoice.tools.weather import get_weather_tool

def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> str:
    """
    Turn on, off, raise, or lower a home automation device.
    
    Args:
        ctx: The run context
        intent: The action to perform (e.g., turn_on, turn_off, raise, lower)
        device: The device to control (e.g., light, fan, shade)
        room: The room where the device is located
    """
    print(f"Home Automation: {intent} {device} in {room}")
    return "Done"
    
class NexusAPIOnline(NexusAPI):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        # TODO: Refactor client ID 
        self.agent = PydanticAgent(config, "TO BE DETERMINED CLIENT ID")
        self.agent.start()
        self.register_tools()

    def register_tools(self):
        self.agent.home_agent.register_tool(home_control)
        self.agent.conversational_agent.register_tool(get_weather_tool)

    def agent_inference(self, agent_id: str, inputs) -> str:
        response = self.agent.process_request(inputs)

        # TODO: Handle ToolCallPart
        response_parts = [part.content for part in response.parts if isinstance(part, TextPart)]

        return "\n".join(response_parts)

    def mcp_agent_inference(self, agent_id: str, inputs) -> ModelResponse:
        assert inputs is not None, "Missing inference input"
        
        if isinstance(inputs, mcp.UserMessage):
            inputs = inputs.text

        print("Input", type(inputs), inputs)
        result: ModelResponse = self.agent.process_request(inputs)
        
        logger.debug(f"Agent Inference Result: \n========\n{result}\n========")

        tool_calls = [call for call in result.parts if isinstance(call, ToolCallPart)]
        if tool_calls:
            tool_results = []
            for call in tool_calls:
                with TimeThis(f"Tool Call: {call.tool_name}"):
                    tool_fn = tool_registry.get(call.tool_name)
                    if tool_fn:
                        output = tool_fn(call.args)
                        logger.debug(f"Tool Call Result: \n========\n{output}\n========")
                        tool_results.append(ToolReturnPart(
                            tool_name=call.tool_name,
                            tool_call_id=call.tool_call_id,
                            content=output
                        ))
                    else:
                        tool_results.append(ToolReturnPart(
                            tool_name=call.tool_name,
                            tool_call_id=call.tool_call_id,
                            content={"error": "Tool not found"}
                        ))

            return self.mcp_agent_tool_response(agent_id, tool_results)
        else:
            return result

    def mcp_agent_tool_response(self, agent_id: str, tool_results: List[ToolReturnPart]) -> ModelResponse:
        print("Tool Results")
        for tool_result in tool_results:
            print("    ", type(tool_result), tool_result)
        assert False, "API not implemented, tool handling should be handled by the agent"
