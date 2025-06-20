"""
Lutron command classes for interacting with Lutron Homeworks system.
"""
from __future__ import annotations
import asyncio
import datetime
import logging
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Generic
)
from enum import Enum, auto

from ..types import (
    ActionT,
    COMMAND_QUERY_PREFIX,
    COMMAND_EXECUTE_PREFIX,
    COMMAND_RESPONSE_PREFIX,
    LINE_END,
    CommandDefinition,
    LutronSpecialEvents,
    CustomHandlerT,
    CommandType,
    CommandError,
    CommandTimeout
)

if TYPE_CHECKING:
    from ..lutron import LutronHomeworksClient
else:
    LutronHomeworksClient = 'LutronHomeworksClient'

from nexusvoice.core.config import load_config
config = load_config()


class CommandResponseProcessors:
    @classmethod
    def passthrough(cls, data: Any) -> Any:
        return data
    
    @classmethod
    def to_int(cls, data: Any) -> int:
        try:
            return int(data)
        except ValueError:
            raise ValueError(f"Invalid integer value: {data}")
    
    @classmethod
    def to_float(cls, data: Any) -> float:
        try:
            return float(data)
        except ValueError:
            raise ValueError(f"Invalid float value: {data}")

    @classmethod
    def to_latlong(cls, data: Any) -> Tuple[float, float]:
        try:
            lat, long = data
            return (float(lat), float(long))
        except ValueError:
            raise ValueError(f"Invalid lat/long value: {data}")
    
    @classmethod
    def to_time(cls, data: Any) -> datetime.time:
        try:
            return datetime.datetime.strptime(data, "%H:%M:%S").time()
        except ValueError:
            raise ValueError(f"Invalid time value: {data}")
    
    @classmethod
    def to_date(cls, data: Any) -> datetime.date:
        try:
            return datetime.datetime.strptime(data, "%m/%d/%Y").date()
        except ValueError:
            raise ValueError(f"Invalid date value: {data}")
    
    @classmethod
    def to_timezone(cls, data: Any) -> datetime.timedelta:
        """Parse a timezone offset string in format [+-]HH:MM and return as timedelta."""
        try:
            # Ensure we have a string
            if not isinstance(data, str):
                data = str(data)
            
            # Handle the sign
            sign = -1 if data.startswith('-') else 1
            
            # Remove the sign if present
            if data.startswith('+') or data.startswith('-'):
                data = data[1:]
                
            # Split hours and minutes
            hours, minutes = map(int, data.split(':'))
            
            # Create a timedelta with the correct sign
            return datetime.timedelta(hours=sign * hours, minutes=sign * minutes)
        except Exception as e:
            raise ValueError(f"Invalid timezone offset value: {data}") from e


class CommandSchema:
    """Schema definition for a command type from a format template."""
    
    def __init__(self, format_template: str, commands: List[CommandDefinition]):
        """
        Define a schema for command parsing and formatting.
        
        Args:
            format_template: Template string with named fields for parsing and formatting
                          e.g., "SYSTEM,{action},{value},{parameters...}"
                          Special field names:
                            - action: The action identifier
                            - value: Primary value
                            - parameters...: Variable number of parameters
                          
                          The command name is extracted from the literal text before the first comma.
        """
        self.format_template: str = format_template
        
        # Extract command name from first part of the template
        parts = format_template.strip().split(',')
        if not parts or not parts[0]:
            raise ValueError("Format template must specify a command before the first comma")
            
        # Store command name from the template
        self._command_name: str = parts[0]
        
        # Parse template to get field positions
        self.response_index_map: Dict[int, str] = self._parse_template(format_template)
        self.commands = {cmd.action: cmd for cmd in commands}
        
    def command_def(self, action: Union[str, Enum]) -> Optional[CommandDefinition]:
        return self.commands.get(action)
    
    def _parse_template(self, template: str) -> Dict[int, str]:
        """Parse a template string to extract field positions."""
        index_map = {}
        
        # Split on commas and extract field names
        parts = template.split(',')
        
        # Skip first part (command name) and start from index 1
        for i, part in enumerate(parts[1:], 1):
            # Extract field name from {field} format
            field = part.strip()
            if field.startswith('{') and field.endswith('}'): 
                field_name = field[1:-1]  # Remove { and }
                
                # Handle parameters... special case
                if field_name.endswith('...'):
                    field_name = field_name[:-3]  # Remove ...
                    index_map[i] = f"{field_name}_start"
                else:
                    index_map[i] = field_name
        
        return index_map
        
    def get_field_index(self, field_name: str) -> Optional[int]:
        """Get the index of a named field in the response."""
        for i, field in self.response_index_map.items():
            if field == field_name:
                return i
        return None

    def get_field_order(self) -> List[str]:
        """Get the order of fields in the response."""
        return [self.response_index_map[key] for key in sorted(self.response_index_map)]
        
    @property
    def command_name(self) -> str:
        """Get the command name extracted from the template."""
        return self._command_name


class UnspecifiedCommandSchema(CommandSchema):
    """A sentinel schema class for unspecified command schemas in base classes."""
    
    def __init__(self):
        """Initialize with dummy template that will throw appropriate errors if used."""
        # Don't actually initialize the parent class
        # Just set up the minimum properties needed to avoid attribute errors
        self.format_template: str = ""
        self.response_index_map: Dict[int, str] = {}
        self._command_name: str = "UNSPECIFIED"
    
    def get_field_index(self, field_name: str) -> Optional[int]:
        """Always return None for unspecified schema."""
        raise NotImplementedError("Command schema unspecified - subclass must define schema")
    
    @property
    def command_name(self) -> str:
        """Return an error message if the schema is unspecified."""
        raise NotImplementedError("Command schema unspecified - subclass must define schema")


class LutronCommand(Generic[ActionT]):
    """Base class for Lutron Homeworks commands."""
    
    # Subclasses should define these as class variables
    schema: ClassVar[CommandSchema]

    response_processors: ClassVar[Dict[Any, Callable[[List[Any]], Any]]] = {}
    
    def __init_subclass__(cls, schema: CommandSchema, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure schema is defined properly
        if isinstance(schema, UnspecifiedCommandSchema):
            raise TypeError("Command schema must be specified for subclass")

        cls.schema = schema
        
    def __init__(self, action: Union[str, ActionT]):
        """
        Initialize base command with action and parameters.
        
        Args:
            action: The action to perform (string or enum value)
            parameters: Optional parameters to include with the command
        """

        self.action = action

        definition = self.schema.command_def(action)
        if definition is None:
            raise ValueError(f"Action {action} not found in schema")
        self.definition: CommandDefinition = definition

        self.processor = self.definition.processor
        self.no_response = self.definition.no_response

        # Default to query/read until otherwise specified.
        if self.definition.is_get:
            self.command_type = CommandType.QUERY
        elif self.definition.is_set:
            self.command_type = CommandType.EXECUTE
        else:
            raise ValueError(f"Action {action} is not a valid get or set action")
        
        self.set_params: Optional[List[Any]] = None
        self._logger = logging.getLogger(self.__class__.__name__)

        self.custom_event: Optional[str] = None
        self.custom_handler: Optional[CustomHandlerT] = None
        

    @property
    def command_name(self) -> str:
        """
        Return the command type string (e.g., "SYSTEM", "OUTPUT").
        
        Uses schema's command_name which is extracted from the format template.
        """
        return self.schema.command_name
    
    @property
    def formatted_command(self) -> str:
        """Format command string from components using the schema."""
        # Choose prefix based on command type
        if self.command_type == CommandType.QUERY:
            prefix = COMMAND_QUERY_PREFIX
        elif self.command_type == CommandType.EXECUTE:
            prefix = COMMAND_EXECUTE_PREFIX
        else:  # Response
            prefix = COMMAND_RESPONSE_PREFIX
            
        # Use schema for formatting - building from left to right
        result = [self.schema.command_name]
        
        all_fields_present = True
        schema_map = self.schema.response_index_map
        ordered_fields = [schema_map[key] for key in sorted(schema_map)]
        for field in ordered_fields:
            if not hasattr(self, field):
                print(f"Field {field} not found")
                all_fields_present = False
                break

            field_value = getattr(self, field)
            result_value = field_value.value if isinstance(field_value, Enum) else field_value
            result.append(result_value)
            
        if all_fields_present and self.set_params:
            print(f"Adding set params: {self.set_params}")
            result.extend(self.set_params)
        
        return f"{prefix}{','.join([str(x) for x in result])}"

    def set(self, set_params: Optional[List[Any]] = None):
        assert self.definition.is_set, f"Action {self.action} is not a valid set action"

        self.command_type = CommandType.EXECUTE
        self.set_params = set_params
        return self
    
    def _matches_response(self, event_data: List[Any]) -> Tuple[bool, List[Any]]:
        """
        Check if the event data matches this command's expected response. Returning
        a tuple of (bool, List[Any]) where the bool indicates if the response matches
        and the List[Any] contains the unmatched data which is indicative of response
        data to be processed.
        
        Args:
            event_data: List of parsed response parts from the Lutron system
                       The first element is the command name (without prefix)
                       The second element is typically the action
                       Additional elements are parameters
        
        Default implementation compares command name and matches each field in the schema
        with values from event_data. Derived classes can override this for special cases.
        """
        # Check for field matches based on schema
        for idx, field_name in enumerate(self.schema.get_field_order()):
            # Skip special fields
            if field_name.endswith('_start'):
                self._logger.warning(f"Skipping special field {field_name}")
                continue
            
            # Get field value from command
            if hasattr(self, field_name):
                attr_value = getattr(self, field_name)
                field_value = attr_value.value if isinstance(attr_value, Enum) else attr_value
            # TODO: check field values from set_params values
            else:
                # All fields in schema matched values present in object
                # Ignoring remaining fields in schema
                print(f"Matches 1: True, Unmatched data: {event_data[idx:]} Event Data: {event_data} Index: {idx}")
                return True, event_data[idx:]
            
            # Match field value with event data
            if str(event_data[idx]) != str(field_value):
                self._logger.warning(f"Field {field_name} does not match: {event_data[idx]} != {field_value}")
                return False, []
        
        # All specified fields match
        print(f"Matches 2: True, Unmatched data: {event_data[len(self.schema.response_index_map):]}")
        return True, event_data[len(self.schema.response_index_map):]

    def handle_response(self, event_data: List[Any], future: asyncio.Future, unsubscribe_func: Callable[[], None]):
        """Handle a response event."""
        # Future is completed (likely by error), ignore response
        if future.done():
            self._logger.debug("Future is done, ignoring response")
            unsubscribe_func()
            return
            
        try:
            # Check if the event matches this command's expected response
            matches, unmatched_data = self._matches_response(event_data)
            if not matches:
                # Not a match for our command
                self._logger.debug(f"Response does not match: {event_data}")
                return
            print(f"Matches: {matches}, Unmatched data: {unmatched_data}")
            # Parse and set response on future
            result = self.process_response(unmatched_data)
            future.set_result(result)
            unsubscribe_func()
        except Exception as e:
            # Set exception on future if parsing fails
            self._logger.exception(f"Error parsing response: {e}")
            future.set_exception(e)
            unsubscribe_func()

    def handle_error(self, event_data: List[Any], future: asyncio.Future, unsubscribe_func: Callable[[], None]):
        """Handle an error event."""
        if future.done():
            self._logger.debug("Future is done, ignoring error")
            unsubscribe_func()
            return
        
        try:
            # Parse error code from data - first element is 'ERROR', second is error code
            if len(event_data) >= 2 and event_data[0] == "ERROR":
                try:
                    error_code = int(event_data[1])
                    future.set_exception(CommandError(error_code, self.formatted_command))
                except (ValueError, IndexError):
                    future.set_exception(CommandError(0, self.formatted_command))
        except Exception as e:
            self._logger.exception(f"Error parsing error: {e}")
            future.set_exception(e)
        finally:
            # Clean up subscriptions
            unsubscribe_func()
            
    def process_response(self, response_data: List[Any]) -> Any:
        """
        Process the response data using the command's schema and response processors.
        
        Args:
            response_data: List of response data from the Lutron system. The first
                       element is the first parameter of the response that does not
                       match a value from the schema that is present in the executed
                       command.
        """ 
        if self.processor is None:
            raise RuntimeError(f"No processor specified for command {self.action}")
        
        # Send response data to processor
        try:
            if len(response_data) == 1:
                processor_args = response_data[0]
            else:
                processor_args = response_data
            print(f"Processor args: {processor_args}")
            return self.processor(processor_args)
        except Exception as e:
            self._logger.exception(f"Error processing response: {e}")
            return response_data

    async def execute(self, lutron_client: LutronHomeworksClient, timeout: float = 5.0):
        """
        Execute the command and return a response.
        
        Args:
            lutron_client: LutronHomeworksClient instance
            timeout: Command timeout in seconds
            
        Returns:
            Response data, parsed according to the particular command
            
        Raises:
            CommandError: If an error is received from Lutron
            CommandTimeout: If the command times out
            ConnectionError: If not connected to Lutron
        """
        # Create a future to track command completion
        future = asyncio.Future()
        
        # Keep track of event subscription tokens for cleanup
        event_tokens = []
        
        # Function to clean up subscriptions
        def unsubscribe_all():
            for token in event_tokens:
                lutron_client.unsubscribe(token)

        formatted_command = self.formatted_command
        
        # Create closures that bind this future and unsubscribe function
        response_handler = lambda event_data: self.handle_response(event_data, future, unsubscribe_all)
        error_handler = lambda event_data: self.handle_error(event_data, future, unsubscribe_all)
        
        # Subscribe to relevant events
        if not self.no_response:
            event_tokens.append(lutron_client.subscribe(self.command_name, response_handler))
        event_tokens.append(lutron_client.subscribe("ERROR", error_handler))
        
        if self.custom_handler is not None:
            custom_handler: CustomHandlerT = self.custom_handler
            subscribe_event = self.command_name
            if self.custom_event:
                subscribe_event = self.custom_event
            wrapper: Callable[[Union[bytes, List[Any]]], None] = lambda param: custom_handler(param, future, unsubscribe_all)
            event_tokens.append(lutron_client.subscribe(subscribe_event, wrapper))
        
        # Send the command and handle any immediate errors
        try:
            await lutron_client.send_command(formatted_command)
        except Exception as e:
            # If command sending fails, set the exception on the future
            import traceback
            traceback.print_exc()
            unsubscribe_all()
            future.set_exception(e)
        
        # Set up timeout handling
        timeout_task = None
        if self.no_response:
            timeout = config.get('tools.lutron.command.no_response_timeout', 0.2)
        
        async def handle_timeout():
            try:
                await asyncio.sleep(timeout)
                if self.no_response:
                    unsubscribe_all()
                    future.set_result(None)
                elif not future.done():
                    unsubscribe_all()
                    future.set_exception(CommandTimeout(f"Command {formatted_command} timed out after {timeout}s"))
            except asyncio.CancelledError:
                pass
        
        # Start timeout monitoring
        timeout_task = asyncio.create_task(
            handle_timeout(),
            name=f"timeout-execute-command"
        )
        
        try:
            # Wait for the future to complete
            result = await future
            # Cancel the timeout task if it's still running
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
            return result
        finally:
            # Ensure timeout task is cancelled
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()