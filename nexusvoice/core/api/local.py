import json
from typing import List
from pydantic_ai.messages import ModelResponsePart, TextPart

from nexusvoice.core.api import NexusAPI
from nexusvoice.core.api.base import ModelResponse
from nexusvoice.core.config import NexusConfig
from nexusvoice.ai.agents import AgentManager
import nexusvoice.protocol.mcp as mcp
from nexusvoice.utils.debug import TimeThis
from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

from nexusvoice.core.api.tool_registry import tool_registry

class NexusAPILocal(NexusAPI):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        self.agent_manager = AgentManager(config)

    def agent_inference(self, agent_id, inputs):
        agent = self.agent_manager.get_agent(agent_id)

        future_result = agent.process_request(inputs)

        # Wait for the result
        result = future_result.result()

        return result
    
    
    def mcp_agent_inference(self, agent_id: str, inputs) -> ModelResponse:
        result_text = self.agent_inference(agent_id, inputs)

        logger.debug(f"Agent Inference Result: \n========\n{result_text}\n========")

        # Try to parse as ToolCall
        try:
            parsed: mcp.ModelMessage = mcp.ModelMessage.model_validate_json(try_repair_json(result_text))
        except Exception:
            return ModelResponse(parts=[TextPart(content=result_text)])

        if parsed.tool_calls:
            tool_results = []
            for call in parsed.tool_calls:
                with TimeThis(f"Tool Call: {call.tool_name}"):
                    tool_fn = tool_registry.get(call.tool_name)
                    if tool_fn:
                        output = tool_fn(call.input)
                        logger.debug(f"Tool Call Result: \n========\n{output}\n========")
                        tool_results.append(mcp.ToolResult(id=call.id, output=output))
                    else:
                        tool_results.append(mcp.ToolResult(id=call.id, output={"error": "Tool not found"}))
            model_message = self.mcp_agent_tool_response(agent_id, tool_results)
            return ModelResponse(parts=[TextPart(content=model_message.text)])

        else:
            return ModelResponse(parts=[TextPart(content=parsed.text)])

    def mcp_agent_tool_response(self, agent_id: str, tool_results: List[mcp.ToolResult]) -> mcp.ModelMessage:
        agent = self.agent_manager.get_agent(agent_id)

        future_result = agent.process_tool_response(tool_results)

        # Wait for the result
        result = future_result.result()

        return result

def try_repair_json(text):
    text = text.strip()

    # Simple fence cleaner
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]

    # Try direct parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as e:
        # Try to patch one missing } at end
        fixed = text + "}"
        try:
            json.loads(fixed)
            return fixed
        except:
            pass

        # Try to close a possibly open string + } at end
        fixed = text + '"}'
        try:
            json.loads(fixed)
            return fixed
        except:
            pass

        raise ValueError(f"Incomplete or invalid JSON: {e}")