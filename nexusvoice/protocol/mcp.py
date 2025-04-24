from abc import abstractmethod, abstractproperty
from pydantic import BaseModel
from typing import Any, ClassVar, List, Optional, Literal, Union, override

MCPMessageType_Type = Literal["tool_call", "tool_result", "render", "user_message", "model_message"]

class MCPMessage(BaseModel):
    """
    Base class for all MCP messages.
    """
    type: MCPMessageType_Type

class ToolCall(MCPMessage):
    type: Literal["tool_call"] = "tool_call" # type: ignore
    tool_name: str
    input: dict
    id: str

class ToolResult(MCPMessage):
    type: Literal["tool_result"] = "tool_result" # type: ignore
    id: str
    output: dict

class Render(MCPMessage):
    type: Literal["render"] = "render" # type: ignore
    content: dict

class UserMessage(MCPMessage):
    type: Literal["user_message"] = "user_message" # type: ignore
    text: str = ""
    attachments: Optional[List[str]] = None

class ModelMessage(MCPMessage):
    type: Literal["model_message"] = "model_message" # type: ignore
    text: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    render: Optional[Render] = None

# Optional: Union for auto-dispatching parsed messages
MCPMessageType = Union[
    UserMessage,
    ModelMessage,
    ToolCall,
    ToolResult,
    Render
]