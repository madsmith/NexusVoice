from pydantic import BaseModel
from typing import List, Optional, Literal, Union

class MCPMessage(BaseModel):
    """
    Base class for all MCP messages.
    """
    type: str

class ToolCall(MCPMessage):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    input: dict
    id: str

class ToolResult(MCPMessage):
    type: Literal["tool_result"] = "tool_result"
    id: str
    output: dict

class Render(MCPMessage):
    type: Literal["render"] = "render"
    content: dict

class UserMessage(MCPMessage):
    type: Literal["user_message"] = "user_message"
    text: Optional[str] = None
    attachments: Optional[List[str]] = None

class ModelMessage(MCPMessage):
    type: Literal["model_message"] = "model_message"
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    render: Optional[Render] = None

# Optional: Union for auto-dispatching parsed messages
MCPMessage = Union[
    UserMessage,
    ModelMessage,
    ToolCall,
    ToolResult,
    Render
]