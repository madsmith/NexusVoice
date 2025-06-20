import asyncio
from datetime import datetime, date
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, List, ClassVar

from nexusvoice.tools.lutron.types import LutronSpecialEvents

from nexusvoice.tools.lutron.types import CommandDefinition as Cmd
from .base import CommandResponseProcessors, LutronCommand, CommandSchema


class SystemAction(Enum):
    """System command actions"""
    TIME = "1"          # Set/Get time
    DATE = "2"          # Set/Get date
    LATLONG = "4"       # Set/Get latitude and longitude
    TIMEZONE = "5"      # Set/Get time zones
    SUNSET = "6"        # Get sunset time
    SUNRISE = "7"       # Get sunrise time
    OS_REV = "8"        # Get OS revision
    LOAD_SHED = "11"    # Set load shed

system_command_definitions = [
    Cmd(SystemAction.TIME, CommandResponseProcessors.to_time),
    Cmd(SystemAction.DATE, CommandResponseProcessors.to_date),
    Cmd(SystemAction.LATLONG, CommandResponseProcessors.to_latlong),
    Cmd(SystemAction.TIMEZONE, CommandResponseProcessors.to_timezone),
    Cmd.GET(SystemAction.SUNSET, CommandResponseProcessors.to_time),
    Cmd.GET(SystemAction.SUNRISE, CommandResponseProcessors.to_time),
    Cmd.GET(SystemAction.OS_REV, CommandResponseProcessors.passthrough),
    Cmd.SET(SystemAction.LOAD_SHED, CommandResponseProcessors.to_int),
]

schema = CommandSchema("SYSTEM,{action}", system_command_definitions)

class SystemCommand(LutronCommand[SystemAction], schema=schema):
    """Command for Lutron Homeworks system-level operations."""
    
    def __init__(self, action: Union[str, SystemAction], parameters: Optional[List[Any]] = None):
        """
        Initialize a system command.
        
        Args:
            action: System action to perform
            parameters: Optional parameters for the command
        """
        # Convert string to enum if needed
        if isinstance(action, SystemAction):
            system_action = action
        else:
            try:
                system_action = SystemAction(action)
            except ValueError:
                raise ValueError(f"Invalid system action: {action}")
                
        super().__init__(action=system_action)

        if self.action == SystemAction.OS_REV:
            self.custom_event = LutronSpecialEvents.NonResponseEvents.value
            self.custom_handler = self._line_handler
    
    # Factory methods for common operations
    @classmethod
    def get_time(cls) -> 'SystemCommand':
        """Get the current system time."""
        return cls(action=SystemAction.TIME)
    
    @classmethod
    def set_time(cls, time_value: str) -> 'SystemCommand':
        """Set the system time.
        
        Args:
            time_value: Time in format SS.ss, SS, MM:SS, or HH:MM:SS
        """
        cmd = cls(action=SystemAction.TIME)
        return cmd.set([time_value])
    
    @classmethod
    def get_date(cls) -> 'SystemCommand':
        """Get the current system date."""
        return cls(action=SystemAction.DATE)
    
    @classmethod
    def set_date(cls, date_value: Union[str, date]) -> 'SystemCommand':
        """Set the system date.
        
        Args:
            date_value: Date in format MM/DD/YYYY or a date object
        """
        if isinstance(date_value, date):
            date_str = date_value.strftime('%m/%d/%Y')
        else:
            date_str = str(date_value)
        cmd = cls(action=SystemAction.DATE)
        return cmd.set([date_str])
    
    @classmethod
    def get_latlong(cls) -> 'SystemCommand':
        """Get the system latitude and longitude."""
        return cls(action=SystemAction.LATLONG)
    
    @classmethod
    def set_latlong(cls, latitude: float, longitude: float) -> 'SystemCommand':
        """Set the system latitude and longitude.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
        """
        lat_long_str = f"{latitude},{longitude}"
        cmd = cls(action=SystemAction.LATLONG)
        return cmd.set([lat_long_str])
    
    @classmethod
    def get_timezone(cls) -> 'SystemCommand':
        """Get the system time zone."""
        return cls(action=SystemAction.TIMEZONE)
    
    @classmethod
    def get_sunset(cls) -> 'SystemCommand':
        """Get the sunset time."""
        return cls(action=SystemAction.SUNSET)
    
    @classmethod
    def get_sunrise(cls) -> 'SystemCommand':
        """Get the sunrise time."""
        return cls(action=SystemAction.SUNRISE)
    
    @classmethod
    def get_os_rev(cls) -> 'SystemCommand':
        """Get the OS revision."""
        return cls(action=SystemAction.OS_REV)
    
    @classmethod
    def set_load_shed(cls, value: Union[bool, int]) -> 'SystemCommand':
        """Set load shed mode.
        
        Args:
            value: True/1 to enable load shed, False/0 to disable
        """
        shed_value = "1" if value else "0"
        cmd = cls(action=SystemAction.LOAD_SHED)
        return cmd.set([shed_value])

    def _line_handler(self, line: Union[bytes, List[Any]], future: asyncio.Future, unsubscribe_all: Callable[[], None]):
        try:
            assert isinstance(line, bytes), "Handler received non-bytes data"
            future.set_result(line.decode('ascii').strip())
        except Exception as e:
            future.set_exception(e)
        finally:
            unsubscribe_all()
