from typing import Annotated, Any, Awaitable, Callable, Literal, TypeVar, Union
from pydantic import BaseModel, Field, model_validator


R = TypeVar('R', covariant=True)

CommandHandlerT = Callable[..., Union[R, Awaitable[R]]]

class CommandParameterError(Exception):
    """Exception raised for errors in command parameters"""
    
    def __init__(self, message: str, missing_params: list[str]  | None = None, invalid_params: dict[str, str] | None = None):
        self.message = message
        self.missing_params = missing_params or []
        self.invalid_params = invalid_params or {}
        
        # Build a detailed error message
        details = [message]
        if self.missing_params:
            details.append(f"Missing parameters: {', '.join(self.missing_params)}")
        if self.invalid_params:
            for param, error in self.invalid_params.items():
                details.append(f"Invalid parameter '{param}': {error}")
                
        super().__init__("\n".join(details))


class CommandDefinition:
    def __init__(
        self, name: str,
        handler: CommandHandlerT,
        params: dict[str, type] | None = None,
        description: str = ""
    ):
        self.name = name
        self.handler = handler
        self.params = params or {}
        self.description = description

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

class CommandParameterInfo(BaseModel):
    type: str
    description: str

class CommandInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, CommandParameterInfo] = Field(default_factory=dict)

class CommandListRequest(BaseModel):
    msg_type: Literal["command_list_request"] = "command_list_request"

class CommandListResponse(BaseModel):
    msg_type: Literal["command_list_response"] = "command_list_response"
    commands: list[CommandInfo]

ServerInboundMessage = Annotated[
    Union[CallRequest],
    Field(discriminator="msg_type")
]

ClientInboundMessage = Annotated[
    Union[
        CallResponseSuccess,
        CallResponseError,
        BroadcastMessage,
        CommandListResponse
    ],
    Field(discriminator="msg_type")
]

Message = Annotated[
    Union[ServerInboundMessage, ClientInboundMessage],
    Field(discriminator="msg_type")
]