
from dataclasses import dataclass, field
from typing import Union
from pydantic import BaseModel, Field

from nexusvoice.core.config import NexusConfig
from pydantic_ai.mcp import MCPServer

@dataclass
class NexusSupportDependencies:
    config: NexusConfig
    servers: dict[str, MCPServer] = field(default_factory=dict)

class RequestType(BaseModel):
    """The type of request being made"""
    type: str = Field(description="The type of request")
    confidence: float = Field(description="Confidence score for the classification")\

class HomeAutomationResponseStruct(BaseModel):
    """Response from the home automation agent"""
    summary_message: str = Field(..., description="A short summary response indicating the success or failure status of the action completed.")

    @staticmethod
    def extract_message(response: "HomeAutomationResponse") -> str:
        if isinstance(response, str):
            return response
        return response.summary_message

HomeAutomationResponse = Union[HomeAutomationResponseStruct, str]

class ConversationResponse(BaseModel):
    """Response from the conversational agent"""
    text: str = Field(..., description="The response text")