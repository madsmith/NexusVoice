from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

class CallRequest(BaseModel):
    msg_type: Literal["call_request"] = "call_request"
    request_id: str
    command: str
    payload: dict

class CallResponse(BaseModel):
    msg_type: Literal["call_response"] = "call_response"
    request_id: str
    status: Literal["ok", "error"]
    result: str

class BroadcastMessage(BaseModel):
    msg_type: Literal["broadcast"] = "broadcast"
    message: str

ServerInboundMessage = Annotated[
    Union[CallRequest],
    Field(discriminator="msg_type")
]

ClientInboundMessage = Annotated[
    Union[CallResponse, BroadcastMessage],
    Field(discriminator="msg_type")
]

Message = Annotated[
    Union[ServerInboundMessage, ClientInboundMessage],
    Field(discriminator="msg_type")
]