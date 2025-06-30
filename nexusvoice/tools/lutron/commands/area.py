from enum import Enum

from typing import Any, List, Optional, Union

from ..types import CommandDefinition as Cmd
from .base import CommandResponseProcessors, LutronCommand, CommandSchema


class AreaAction(Enum):
    LEVEL = "1"            # Set area level
    START_RAISE = "2"      # Start raising area level
    START_LOWER = "3"      # Start lowering area level
    STOP_RAISE_LOWER = "4" # Stop raising/lowering
    SCENE = "6"            # Set/Get current scene
    OCCUPANCY_STATE = "8"  # Get occupancy state
    
area_command_definitions = [
    # TODO: Confirm action with outputs from area
    Cmd.SET(AreaAction.LEVEL, CommandResponseProcessors.to_float, no_response=True),
    Cmd.SET(AreaAction.START_RAISE, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(AreaAction.START_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd.SET(AreaAction.STOP_RAISE_LOWER, CommandResponseProcessors.to_int, no_response=True),
    Cmd(AreaAction.SCENE, CommandResponseProcessors.to_int),
    Cmd.GET(AreaAction.OCCUPANCY_STATE, CommandResponseProcessors.to_int),
]

schema = CommandSchema("AREA,{iid},{action}", area_command_definitions)


class AreaCommand(LutronCommand[AreaAction], schema=schema):

    def __init__(self, iid: int, action: Union[str, AreaAction]):
        """
        Initialize an area command.
        
        Args:
            iid: Integration ID of the area
            action: Area action to perform
        """
        # Convert string to enum if needed
        if isinstance(action, AreaAction):
            area_action = action
        else:
            try:
                area_action = AreaAction(action)
            except ValueError:
                raise ValueError(f"Invalid area action: {action}")

        super().__init__(action=area_action)

        self.iid = iid

    @classmethod
    def set_level(cls, iid: int, level: float) -> 'AreaCommand':
        """Set the area level (0.0-100.0)."""
        return cls(iid, AreaAction.LEVEL).set([level])

    @classmethod
    def start_raise(cls, iid: int) -> 'AreaCommand':
        """Start raising the area level."""
        return cls(iid, AreaAction.START_RAISE)
    
    @classmethod
    def start_lower(cls, iid: int) -> 'AreaCommand':
        """Start lowering the area level."""
        return cls(iid, AreaAction.START_LOWER)
    
    @classmethod
    def stop_raise_lower(cls, iid: int) -> 'AreaCommand':
        """Stop raising/lowering the area level."""
        return cls(iid, AreaAction.STOP_RAISE_LOWER)
    
    @classmethod
    def set_scene(cls, iid: int, scene: int) -> 'AreaCommand':
        """Set the current scene for the area."""
        return cls(iid, AreaAction.SCENE).set([scene])
    
    @classmethod
    def get_scene(cls, iid: int) -> 'AreaCommand':
        """Get the current scene for the area."""
        return cls(iid, AreaAction.SCENE)
    
    @classmethod
    def get_occupancy_state(cls, iid: int) -> 'AreaCommand':
        """Get the current occupancy state for the area.
        
        Returns:
            1: Unknown State
            2: Inactive
            3: Occupied
            4: Unoccupied
        """
        return cls(iid, AreaAction.OCCUPANCY_STATE)
