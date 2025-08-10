#!/usr/bin/env python3
import asyncio
import json
import logfire
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter
from typing import Any, Callable, Protocol, Union, Awaitable
import uuid

from nexusvoice.utils.eventbus import EventBus, SubscriptionToken

from .types import (
    CallRequest,
    CallResponse,
    CallResponseSuccess,
    CallResponseError,
    ServerMessage,
    ClientInboundMessage,
    CommandInfo,
    CommandListResponse,
)

message_adapter = TypeAdapter(ClientInboundMessage)

CallbackT = Callable[[Any], Union[Any, Awaitable[Any]]]

class NexusConnection:
    """
    A bi-directional connection to the NexusServer that handles both sending
    commands and asynchronously receiving messages from the server.
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

        self.connection_id: uuid.UUID | None = None
        self.call_id: int = 0
        self.pending_calls: dict[str, asyncio.Future] = {}
        self.lock = asyncio.Lock()

        self._event_bus: EventBus = EventBus()

        self.running = False
        self.read_task = None
    
    async def connect(self) -> bool:
        """
        Connect to the NexusServer.
        
        Returns:
            Boolean indicating if the connection was successful
            
        Raises:
            ConnectionRefusedError: If the server is not available at the specified host/port
            OSError: If there is a network or system-level error during connection
        """
        try:
            logfire.info(f"Connecting to server at {self.host}:{self.port}...")
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self.connection_id = uuid.uuid4()
            self.call_id = 0
            logfire.info("Connected successfully!")
            
            # Start the message reading task
            self.read_task = asyncio.create_task(self._read_messages())
            
            return True
        except (ConnectionRefusedError, OSError) as e:
            logfire.error(f"Connection refused. Is the server running at {self.host}:{self.port}?")
        except Exception as e:
            logfire.error(f"Connection error: {e}")
        
        # Connection failed
        self.connection_id = None
        self.call_id = 0
        return False
    
    async def disconnect(self) -> None:
        """
        Disconnect from the server and clean up resources.
        
        This method cancels any pending read tasks and properly closes the writer.
        It is safe to call even if not connected.
        """
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass
            self.read_task = None
            
            try:
                async with self.lock:
                    if self.writer:
                        self.writer.close()
                        await self.writer.wait_closed()
                        self.writer = None
                        logfire.info("Disconnected from server")
            except Exception as e:
                error_msg = f"Error during disconnect: {e}"
                logfire.error(error_msg)
        
        self.connection_id = None
        self.call_id = 0
    
    @property
    def connected(self):
        """
        Check if we are connected to the server.
        
        Returns:
            Boolean indicating if currently connected to the server
        """
        return self.connection_id is not None
    
    def get_task(self):
        """
        Get the message reading task.
        
        Returns:
            The asyncio Task handling background message processing, or None if not connected
        """
        return self.read_task

    def subscribe(self, event_type: str, callback: CallbackT) -> SubscriptionToken:
        """
        Subscribe to events of a specific type with either a sync or async callback function.
        
        Args:
            event_type: The type of event to subscribe to
            callback: A callback function that can be either synchronous or asynchronous
            
        Returns:
            A subscription token that can be used to unsubscribe later
        """
        return self._event_bus.subscribe(event_type, callback)

    def on_server_message(self, callback: CallbackT) -> SubscriptionToken:
        """
        Shorthand to subscribe to server messages.
        
        Args:
            callback: A callback function that will be called when server messages are received
            
        Returns:
            A subscription token that can be used to unsubscribe later
        """
        return self.subscribe("server_message", callback)
    
    def unsubscribe(self, token: SubscriptionToken):
        """
        Unsubscribe from an event using the token returned by subscribe.
        
        Args:
            token: The subscription token returned by a previous call to subscribe
        """
        self._event_bus.unsubscribe(token)
    
    async def _read_messages(self):
        """Background task to continuously read messages from the server"""
        try:
            while self.connected and self.reader:
                # Read a line from the server
                data = await self.reader.readline()
                
                if not data:  # Connection closed
                    logfire.info("\nServer closed the connection")
                    break
                    
                # Process the message
                try:
                    await self._process_server_message(data)
                except json.JSONDecodeError:
                    logfire.error(f"Received invalid JSON: {data.decode().strip()}")
                except Exception as e:
                    logfire.error(f"Error processing message: {e}")
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        except Exception as e:
            error_msg = f"Error in message reading task: {e}"
            logfire.error(error_msg)
        finally:
            await self.disconnect()
            
    async def _process_server_message(self, data: bytes):
        """Process a message received from the server"""
        try:
            msg = message_adapter.validate_json(data)
        except ValidationError as e:
            error_msg = f"Invalid message from server: {data.decode(errors='replace')} :: {e}"
            logfire.error(error_msg)
            return

        if isinstance(msg, CallResponse):  # Use parent class directly for type checking
            await self._process_call_response(msg)
        elif isinstance(msg, ServerMessage):
            response_msg = msg.message
            logfire.debug(f"Server Message: {response_msg}")
            # Emit event for server messages
            await self._event_bus.emit('server_message', msg.message)
        else:
            warning_msg = f"Unhandled message type: [{type(msg)}] {msg}"
            logfire.warning(warning_msg)
    
    async def _process_call_response(self, response: CallResponse):
        """
        Process a response from the server, matching the request ID to the pending call
        and setting the pending call result to the waiting future.
        """
        # Look for a pending request with matching ID
        if response.request_id in self.pending_calls:
            future = self.pending_calls.get(response.request_id)
            if future and not future.done():
                future.set_result(response)
                return True
        return False
    
    async def send_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = 60.0
    ) -> Any:
        """
        Send a command to the server and wait for a response.
        
        Args:
            command: The command name to execute on the server
            payload: Optional dictionary of parameters to pass to the command
            timeout: Maximum time in seconds to wait for a response, or None for no timeout
            
        Returns:
            The result returned by the server command
            
        Raises:
            ConnectionError: If not connected to the server when attempting to send
            asyncio.TimeoutError: If the server does not respond within the timeout period
            Exception: If the server returns an error response or if there's an issue with sending
                the command or processing the response
        """
        if not self.connected or not self.writer:
            logfire.error("Send Command Error: Not connected to server")
            return
        assert self.connection_id is not None

        if payload is None:
            payload = {}
        
        call_name = f"call_{self.call_id}"
        self.call_id += 1
        request_id = str(uuid.uuid5(self.connection_id, call_name))

        call_request = CallRequest(
            request_id=request_id,
            command=command,
            payload=payload
        )
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        try:
            # Send command to server
            data = call_request.model_dump_json().encode() + b'\n'
            async with self.lock:
                self.pending_calls[request_id] = future
                self.writer.write(data)
                await self.writer.drain()
        except Exception as e:
            logfire.error(f"Error sending command: {e}")
            self.pending_calls.pop(request_id, None)
            await self.disconnect()
            raise

        try:
            result = await asyncio.wait_for(future, timeout)
            if isinstance(result, CallResponseError):
                logfire.error(f"Command {command} failed: {result.error}")
                raise Exception(result.error)
            elif isinstance(result, CallResponseSuccess):
                return result.result
            else:
                logfire.error(f"Received invalid response from server: {result} [{type(result)}]")
                raise Exception("Invalid response from server")
        except asyncio.TimeoutError:
            logfire.error(f"Command {command} timed out after {timeout} seconds")
            future.cancel()
            raise
        finally:
            self.pending_calls.pop(request_id, None)

    async def list_commands(self) -> list[CommandInfo]:
        """
        Retrieve a list of available commands from the server.
        
        Returns:
            List of CommandInfo objects describing available server commands
            
        Raises:
            ValidationError: If the server response cannot be parsed as a command list
            Exceptions from send_command: If there are connection or timeout errors
        """
        response = await self.send_command("list_commands")

        try:
            command_list = CommandListResponse.model_validate(response)
            return command_list.commands
        except ValidationError as e:
            logfire.error(f"List Commands - Invalid Response: {response}")
            return []
        