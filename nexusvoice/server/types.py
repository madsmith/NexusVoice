from typing import Literal, Union, Annotated, Any
from pydantic import BaseModel, Field, model_validator

class CallRequest(BaseModel):
    msg_type: Literal["call_request"] = "call_request"
    request_id: str
    command: str
    payload: dict

class CallResponse(BaseModel):
    """Base class for all response types"""
    request_id: str

    @model_validator(mode="before")
    @classmethod
    def prevent_instantiation(cls, data):
        if cls is CallResponse:
            raise TypeError("CallResponse must not be instantiated directly")
        return data

class CallResponseSuccess(CallResponse):
    msg_type: Literal["call_response_success"] = "call_response_success"
    result: Any

class CallResponseError(CallResponse):
    msg_type: Literal["call_response_error"] = "call_response_error"
    error: str
    details: dict = Field(default_factory=dict)

class BroadcastMessage(BaseModel):
    msg_type: Literal["broadcast"] = "broadcast"
    message: str

ServerInboundMessage = Annotated[
    Union[CallRequest],
    Field(discriminator="msg_type")
]

ClientInboundMessage = Annotated[
    Union[CallResponseSuccess, CallResponseError, BroadcastMessage],
    Field(discriminator="msg_type")
]

Message = Annotated[
    Union[ServerInboundMessage, ClientInboundMessage],
    Field(discriminator="msg_type")
]