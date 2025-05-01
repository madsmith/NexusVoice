
from dataclasses import dataclass

from nexusvoice.core.config import NexusConfig
from pydantic import BaseModel, Field

@dataclass
class NexusSupportDependencies:
    config: NexusConfig

class RequestType(BaseModel):
    """The type of request being made"""
    type: str = Field(description="The type of request")
    confidence: float = Field(description="Confidence score for the classification")