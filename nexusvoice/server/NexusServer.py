import asyncio
import importlib
import inspect
import json
import logfire
import pkgutil
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

from .types import (
    BroadcastMessage,
    CallRequest,
    CallResponse,
    CallResponseError,
    CallResponseSuccess,
    CommandDefinition,
    CommandHandlerT,
    CommandInfo,
    CommandListResponse,
    CommandParameterError,
    CommandParameterInfo,
    ServerInboundMessage,
)
from .tasks.base import NexusTask
from .registry import ServiceRegistry

from nexusvoice.core.config import NexusConfig

inbound_message_adapter = TypeAdapter(ServerInboundMessage)


class NexusServer:
    def __init__(self, config: NexusConfig):
        self.config = config
        self.host = config.get("nexus.server.host", "localhost")
        self.port = config.get("nexus.server.port", 8000)

        self.server = None
        self.server_socket = None
        self.clients: dict[str, asyncio.StreamWriter] = {}
        self.running = False

        self.tasks: list[NexusTask] = []
        self.service_registry = ServiceRegistry()

        self.lock = asyncio.Lock()
        
        # Command handlers
        self.command_registry: dict[str, CommandDefinition] = {}

        self.register_command(
            "list_commands",
            self._handle_list_commands,
            params=None,
            description="List available commands"
        )
        self.register_command("ping", self._handle_ping, params=None, description="Ping the server")
        self.register_command(
            "queue_broadcast",
            self._handle_queue_broadcast,
            params=None,
            description="Queue a broadcast message"
        )

    async def start(self):
        """Initialize the server socket and start listening for connections"""
        # Discover and initialize tasks
        await self._load_tasks()
        
        logfire.info(f"Starting NexusServer on {self.host}:{self.port}")
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )

        self.running = True
        
        async with self.server:
            await self._main_loop()

    async def stop(self):
        """Stop the server and close all connections"""
        self.running = False
        
        async with self.lock:
            # Close all client connections
            for client_id, writer in self.clients.items():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    # Don't allow inspection during shutdown, format message manually
                    error_msg = f"Error closing connection for {client_id}: {e}"
                    logfire.error(error_msg)
        
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
        
        logfire.info("Server stopped")

    def register_command(
        self, command: str, handler: CommandHandlerT,
        params: dict[str, type] | None = None,
        description: str = ""
    ):
        """Register a command handler"""
        if command in self.command_registry:
            raise ValueError(f"Command {command} is already registered")
        self.command_registry[command] = CommandDefinition(
            command, handler, params, description
        )
        
    @logfire.instrument("Loading tasks")
    async def _load_tasks(self):
        """Discover, instantiate, and register NexusTask classes"""
        import nexusvoice.server.tasks as tasks_module
        
        # Find all submodules in the tasks package
        with logfire.span("Discovering tasks"):
            for _, name, is_pkg in pkgutil.iter_modules(tasks_module.__path__, tasks_module.__name__ + "."):
                if not is_pkg and name != tasks_module.__name__ + ".base":
                    try:
                        # Import the module
                        module = importlib.import_module(name)
                    
                        # Find all NexusTask subclasses in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (inspect.isclass(attr) and issubclass(attr, NexusTask) and attr != NexusTask):
                                # Instantiate the task
                                task = attr(self, self.config)
                                logfire.info(f"Discovered task: {attr.__name__}")
                                self.tasks.append(task)
                    except Exception as e:
                        logfire.error(f"Error loading task module {name}: {e}")
        
        # Register all tasks
        with logfire.span("Registering tasks"):
            failed_tasks = []
            for task in self.tasks:
                try:
                    task.register()
                    logfire.info(f"Registered task: {task.__class__.__name__}")
                except Exception as e:
                    logfire.error(f"Error registering task {task.__class__.__name__}: {e}")
                    failed_tasks.append(task)

            # Remove failed tasks
            self.tasks = [task for task in self.tasks if task not in failed_tasks]

        # Initialize tasks
        with logfire.span("Initializing tasks"):
            initialized_tasks = []
            for task in self.tasks:
                try:
                    with logfire.span(f"Initializing task: {task.__class__.__name__}"):
                        success = await task.initialize()
                        if success:
                            initialized_tasks.append(task)
                        else:
                            logfire.error(f"Task {task.__class__.__name__} failed to initialize")
                except Exception as e:
                    logfire.error(f"Error initializing task {task.__class__.__name__}: {e}")
            
            # Replace tasks list with only successfully initialized tasks
            self.tasks = initialized_tasks
                
    async def _stop_tasks(self):
        """Stop all running tasks"""
        for task in self.tasks:
            try:
                await task.stop()
                logfire.info("Stopped task: " + task.__class__.__name__)
            except Exception as e:
                logfire.error(f"Error stopping task {task.__class__.__name__}: {e}")

    @logfire.instrument("Server Main Loop")
    async def _main_loop(self):
        """Main server loop"""
        # Start task coroutines but don't await them yet
        task_coros = [task.start() for task in self.tasks]
        tasks = []
        
        try:
            # Start all tasks
            for coro in task_coros:
                tasks.append(asyncio.create_task(coro))
                
            # Keep the server running until it is cancelled
            await asyncio.Future()
        except asyncio.CancelledError:
            logfire.info("Server loop cancelled")
        finally:
            # Stop all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to be cancelled
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Call stop on all task objects
            await self._stop_tasks()

            # Close the server
            await self.stop()
    
    @logfire.instrument("Handle Client Connection")
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection"""
        # Generate a unique client ID   
        client_address = writer.get_extra_info('peername')
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        # Store the client connection
        async with self.lock:
            self.clients[client_id] = writer
        logfire.info(f"New client connected: {client_id}")
        
        try:
            while self.running:
                # Read data from client
                data = await reader.readline()
                if not data:
                    break
                    
                # Process the received command
                await self._process_command(client_id, data, writer)
        
        except asyncio.CancelledError:
            logfire.info(f"Client handler for {client_id} cancelled")
        except Exception as e:
            logfire.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up when client disconnects
            async with self.lock:
                writer.close()
                await writer.wait_closed()
                if client_id in self.clients:
                    del self.clients[client_id]
            logfire.info(f"Client disconnected: {client_id}")
    
    @logfire.instrument("Process Command")
    async def _process_command(self, client_id: str, data: bytes, writer: asyncio.StreamWriter):
        try:
            try:
                msg = inbound_message_adapter.validate_json(data)
            except ValidationError as e:
                logfire.error(f"Invalid message from client {client_id}: {data.decode(errors='replace')} :: {e}")
                error_response = CallResponseError(
                    request_id="ERROR",
                    error="Invalid JSON format",
                    details={"raw_data": data.decode(errors='replace')}
                )
                await self._send_response(writer, error_response)
                return

            if isinstance(msg, CallRequest):
                command_definition = self.command_registry.get(msg.command)
                if command_definition:
                    try:
                        # Marshal arguments from payload based on command definition
                        args = self._marshal_arguments(msg.command, command_definition, msg.payload)
                        
                        # Call the handler with the marshaled arguments
                        result = await command_definition.handler(client_id, **args)
                        response = CallResponseSuccess(
                            request_id=msg.request_id,
                            result=result
                        )
                    except CommandParameterError as e:
                        # Handle parameter validation errors
                        logfire.info(f"Parameter validation error for command {msg.command}: {e.message}")
                        response = CallResponseError(
                            request_id=msg.request_id,
                            error="Invalid command parameters",
                            details={
                                "message": e.message,
                                "missing_parameters": e.missing_params,
                                "invalid_parameters": e.invalid_params
                            }
                        )
                    except Exception as e:
                        logfire.error(f"Error executing command {msg.command}: {e}")
                        response = CallResponseError(
                            request_id=msg.request_id,
                            error=f"Command execution error",
                            details={"exception": str(e)}
                        )
                else:
                    response = CallResponseError(
                        request_id=msg.request_id,
                        error=f"Unknown command",
                        details={"command": msg.command}
                    )
                await self._send_response(writer, response)

        except Exception as e:
            logfire.error(f"Unexpected error in _process_command: {e}")
    
    def _marshal_arguments(self, command_name: str, command_def: CommandDefinition, payload: dict) -> dict:
        """
        Convert and validate arguments from the payload according to command parameter specifications.
        
        Args:
            command_name: The name of the command being executed
            command_def: The command definition with parameter types
            payload: The raw payload from the client
            
        Returns:
            A dictionary of validated and converted arguments
            
        Raises:
            CommandParameterError: If parameters are missing or invalid
        """
        if not command_def.params:
            # No parameters defined for this command, return payload as-is
            return payload or {}
        
        # Initialize arguments dictionary
        args = {}
        missing_params = []
        invalid_params = {}
        
        # Check for each defined parameter
        for param_name, param_type in command_def.params.items():
            if param_name not in payload:
                missing_params.append(param_name)
                continue
                
            # Get the value from the payload
            value = payload[param_name]
            
            # Try to convert/validate the value
            try:
                # Handle basic types
                if param_type == str:
                    args[param_name] = str(value)
                elif param_type == int:
                    args[param_name] = int(value)
                elif param_type == float:
                    args[param_name] = float(value)
                elif param_type == bool:
                    # Support various boolean representations
                    if isinstance(value, bool):
                        args[param_name] = value
                    elif isinstance(value, str):
                        args[param_name] = value.lower() in ('true', 'yes', '1', 'y')
                    elif isinstance(value, (int, float)):
                        args[param_name] = bool(value)
                    else:
                        invalid_params[param_name] = f"Cannot convert {type(value).__name__} to bool"
                else:
                    # For other types, just pass through and hope for the best
                    # In a more robust system, we would use a more sophisticated validation system
                    args[param_name] = value
            except (ValueError, TypeError) as e:
                invalid_params[param_name] = str(e)
        
        # Check if there were any issues
        if missing_params or invalid_params:
            raise CommandParameterError(
                message=f"Invalid parameters for command '{command_name}'", 
                missing_params=missing_params, 
                invalid_params=invalid_params
            )
            
        return args
    
    @logfire.instrument("Send Response")
    async def _send_response(self, writer: asyncio.StreamWriter, response: CallResponse | BroadcastMessage):
        try:
            response_data = response.model_dump_json().encode() + b'\n'
            writer.write(response_data)
            await writer.drain()
        except Exception as e:
            logfire.error(f"Error sending response: {e}")

    async def _handle_list_commands(self, client_id: str) -> CommandListResponse:
        """
        Present a list of available commands to the client
        """
        commands = []
        for command in self.command_registry.values():
            params = {}
            for param_name, param_type in command.params.items():
                params[param_name] = CommandParameterInfo(
                    type=param_type.__name__,
                    description=f"{param_name}: {param_type.__name__}"
                )
            commands.append(CommandInfo(
                name=command.name,
                description=command.description,
                parameters=params
            ))
        return CommandListResponse(commands=commands)
        
    async def _handle_ping(self, client_id: str) -> str:
        """Handle ping command from client"""
        logfire.info(f"Ping from {client_id}")
        return "pong"

    async def _handle_queue_broadcast(self, client_id: str) -> str:
        """Handle queue broadcast command from client"""
        logfire.info(f"Queue broadcast from {client_id}")
        # defer queue a broadcast message in 2 seconds
        async def do_broadcast():
            await asyncio.sleep(4)
            await self.broadcast(BroadcastMessage(message="Hello from NexusServer"))

        asyncio.create_task(do_broadcast())
        
        return "ok"
    
    async def broadcast(self, message: BroadcastMessage):
        """Broadcast a message to all connected clients"""
        async with self.lock:
            for client_id, writer in self.clients.items():
                asyncio.create_task(self._send_response(writer, message))