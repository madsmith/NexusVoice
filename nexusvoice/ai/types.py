
from dataclasses import dataclass
from typing import Union
from pydantic import BaseModel, Field

from nexusvoice.core.config import NexusConfig

@dataclass
class NexusSupportDependencies:
    config: NexusConfig

class RequestType(BaseModel):
    """The type of request being made"""
    type: str = Field(description="The type of request")
    confidence: float = Field(description="Confidence score for the classification")

class HomeAutomationAction(BaseModel):
    intent: str = Field(..., description="The action to perform (e.g., turn_on, turn_off, raise, lower)")
    device: str = Field(..., description="The device to control (e.g., light, fan, shade)")
    room: str = Field(..., description="The room where the device is located")

class HomeAutomationResponseStruct(BaseModel):
    """Response from the home automation agent"""
    summary_message: str = Field(..., description="A short summary response indicating the success or failure status of the action completed.")

    @staticmethod
    def extract_message(response: "HomeAutomationResponse") -> str:
        if isinstance(response, str):
            return response
        return response.summary_message

HomeAutomationResponse = Union[HomeAutomationResponseStruct, str]