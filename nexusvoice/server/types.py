from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, TypeAdapter

class CallRequest(BaseModel):
    msg_type: Literal["call_request"] = "call_request"
    request_id: str
    command: str
    payload: dict

class CallResponse(BaseModel):
    msg_type: Literal["call_response"] = "call_response"
    request_id: str
    status: str  # Consider Literal["ok", "error"]
    result: str

class BroadcastMessage(BaseModel):
    msg_type: Literal["broadcast"] = "broadcast"
    message: str

Message = Annotated[
    Union[CallRequest, CallResponse, BroadcastMessage],
    Field(discriminator="msg_type")
]

message_adapter = TypeAdapter(Message)