from enum import Enum

from typing import Any, List, Optional, Union

from ..types import CommandDefinition as Cmd
from .base import CommandResponseProcessors, LutronCommand, CommandSchema

class ShadeGroupAction(Enum):
    ZONE_LEVEL = "1"         # Set/Get Zome Level
    START_RAISE = "2"        # Set Start raising
    START_LOWER = "3"        # Set Start lowering
    STOP_RAISE_LOWER = "4"   # Set Stop raising/lowering
    CURRENT_PRESET = "6"     # Set Current preset
    
shade_group_command_definitions = [
    Cmd(ShadeGroupAction.ZONE_LEVEL, CommandResponseProcessors.to_float),
    Cmd.SET(ShadeGroupAction.START_RAISE, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(ShadeGroupAction.START_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(ShadeGroupAction.STOP_RAISE_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(ShadeGroupAction.CURRENT_PRESET, CommandResponseProcessors.to_int),
]

schema = CommandSchema("SHADEGRP,{iid},{action}", shade_group_command_definitions)

class ShadeGroupCommand(LutronCommand[ShadeGroupAction], schema=schema):

    def __init__(self, iid: int, action: Union[str, ShadeGroupAction]):
        """
        Initialize an output command.
        
        Args:
            action: Output action to perform
            parameters: Optional parameters for the command
        """
        # Convert string to enum if needed
        if isinstance(action, ShadeGroupAction):
            shade_group_action = action
        else:
            try:
                shade_group_action = ShadeGroupAction(action)
            except ValueError:
                raise ValueError(f"Invalid output action: {action}")

        super().__init__(action=shade_group_action)

        self.iid = iid

    @classmethod
    def get_zone_level(cls, iid: int) -> 'ShadeGroupCommand':
        """Get the current zone level."""
        return cls(iid, ShadeGroupAction.ZONE_LEVEL)
    
    @classmethod
    def set_zone_level(cls, iid: int, level: float) -> 'ShadeGroupCommand':
        """Set the zone level."""
        cmd = cls(iid, ShadeGroupAction.ZONE_LEVEL)
        return cmd.set([level])

    @classmethod
    def start_raise(cls, iid: int) -> 'ShadeGroupCommand':
        """Start raising the zone."""
        return cls(iid, ShadeGroupAction.START_RAISE)
    
    @classmethod
    def start_lower(cls, iid: int) -> 'ShadeGroupCommand':
        """Start lowering the zone."""
        return cls(iid, ShadeGroupAction.START_LOWER)
    
    @classmethod
    def stop_raise_lower(cls, iid: int) -> 'ShadeGroupCommand':
        """Stop raising/lowering the zone."""
        return cls(iid, ShadeGroupAction.STOP_RAISE_LOWER)
    
    @classmethod
    def get_current_preset(cls, iid: int) -> 'ShadeGroupCommand':
        """Get the current preset."""
        return cls(iid, ShadeGroupAction.CURRENT_PRESET)