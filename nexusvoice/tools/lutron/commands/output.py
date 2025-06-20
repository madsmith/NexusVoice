from enum import Enum

from typing import Any, List, Optional, Union

from ..types import CommandDefinition as Cmd
from .base import CommandResponseProcessors, LutronCommand, CommandSchema

class OutputAction(Enum):
    ZONE_LEVEL = "1"         # Set/Get Zome Level
    START_RAISE = "2"        # Set Start raising
    START_LOWER = "3"        # Set Start lowering
    STOP_RAISE_LOWER = "4"   # Set Stop raising/lowering
    PULSE_TIME = "5"         # Set Pulse time
    
output_command_definitions = [
    Cmd(OutputAction.ZONE_LEVEL, CommandResponseProcessors.to_float),
    Cmd.SET(OutputAction.START_RAISE, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(OutputAction.START_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(OutputAction.STOP_RAISE_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(OutputAction.PULSE_TIME, CommandResponseProcessors.to_int, no_response=True),
]

schema = CommandSchema("OUTPUT,{iid},{action}", output_command_definitions)

class OutputCommand(LutronCommand[OutputAction], schema=schema):

    def __init__(self, iid: int, action: Union[str, OutputAction], parameters: Optional[List[Any]] = None):
        """
        Initialize an output command.
        
        Args:
            action: Output action to perform
            parameters: Optional parameters for the command
        """
        # Convert string to enum if needed
        if isinstance(action, OutputAction):
            output_action = action
        else:
            try:
                output_action = OutputAction(action)
            except ValueError:
                raise ValueError(f"Invalid output action: {action}")

        super().__init__(action=output_action)

        self.iid = iid

    @classmethod
    def get_zone_level(cls, iid: int) -> 'OutputCommand':
        """Get the current zone level."""
        return cls(iid, OutputAction.ZONE_LEVEL)
    
    @classmethod
    def set_zone_level(cls, iid: int, level: float) -> 'OutputCommand':
        """Set the zone level."""
        cmd = cls(iid, OutputAction.ZONE_LEVEL)
        return cmd.set([level])

    @classmethod
    def start_raise(cls, iid: int) -> 'OutputCommand':
        """Start raising the zone."""
        return cls(iid, OutputAction.START_RAISE)
    
    @classmethod
    def start_lower(cls, iid: int) -> 'OutputCommand':
        """Start lowering the zone."""
        return cls(iid, OutputAction.START_LOWER)
    
    @classmethod
    def stop_raise_lower(cls, iid: int) -> 'OutputCommand':
        """Stop raising/lowering the zone."""
        return cls(iid, OutputAction.STOP_RAISE_LOWER)
    
    @classmethod
    def set_pulse_time(cls, iid: int, pulse_time: int) -> 'OutputCommand':
        """Set the pulse time."""
        return cls(iid, OutputAction.PULSE_TIME, [pulse_time])