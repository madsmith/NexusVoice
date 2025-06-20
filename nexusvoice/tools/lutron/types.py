import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Generic, List, Optional, TypeVar, Union

# Command prefixes
COMMAND_QUERY_PREFIX = "?"
COMMAND_EXECUTE_PREFIX = "#"
COMMAND_RESPONSE_PREFIX = "~"


class CommandType(Enum):
    QUERY = auto()
    EXECUTE = auto()
    RESPONSE = auto()

LINE_END = "\r\n"

class LutronSpecialEvents(Enum):
    AllEvents = "::[*]::"
    NonResponseEvents = "::[msg]::"

# Define a type variable for action enums
ActionT = TypeVar('ActionT', bound=Union[str, Enum])

# Custom handler type for command response processing
CustomHandlerT = Callable[[Union[bytes, List[Any]], asyncio.Future, Callable[[], None]], None]

@dataclass
class CommandDefinition:
    action: Union[str, Enum]
    processor: Optional[Callable[[Any], Any]] = None
    no_response: bool = False
    is_get: bool = True
    is_set: bool = True

    @classmethod
    def _make_command(cls, action: Union[str, Enum], processor: Optional[Callable[[Any], Any]] = None, **kwargs):
        return CommandDefinition(action=action, **kwargs)
    
    @classmethod
    def GET(cls, action: Union[str, Enum], processor: Optional[Callable[[Any], Any]] = None, **kwargs):
        return cls._make_command(action, is_set=False, processor=processor, **kwargs)
    
    @classmethod
    def SET(cls, action: Union[str, Enum], processor: Optional[Callable[[Any], Any]] = None, **kwargs):
        return cls._make_command(action, is_get=False, processor=processor, **kwargs)
    
    @classmethod
    def GETSET(cls, action: Union[str, Enum], processor: Optional[Callable[[Any], Any]] = None, **kwargs):
        return cls._make_command(action, is_get=True, is_set=True, **kwargs)
    


# Error types
class LutronError(Exception):
    """Base class for Lutron-related errors."""
    pass


class CommandError(LutronError):
    """Error raised when a command fails."""
    ERROR_MESSAGES = {
        1: "Parameter count mismatch",
        2: "Object does not exist",
        3: "Invalid action number",
        4: "Parameter data out of range",
        5: "Parameter data malformed",
        6: "Unsupported Command"
    }

    def __init__(self, error_code: int, command: Optional[str] = None):
        self.error_code = error_code
        self.command = command
        message = self.ERROR_MESSAGES.get(error_code, f"Unknown error: {error_code}")
        if command:
            message = f"{message} (command: {command})"
        super().__init__(message)


class CommandTimeout(LutronError):
    """Error raised when a command times out."""
    pass
