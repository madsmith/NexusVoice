#!/usr/bin/env python3
import argparse
import asyncio
import os
import shlex
import logfire
import logging
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.patch_stdout import patch_stdout
from typing import Iterable

from nexusvoice.bootstrap import get_logfire

from .connection import NexusConnection
from .types import CommandInfo

logger = logging.getLogger(__name__)

class CommandCompleter(Completer):
    def __init__(self, repl_commands: list[str], command_map: dict[str, CommandInfo]):
        self.repl_commands = repl_commands
        self.command_map = command_map
    
    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text
        for command in self.repl_commands:
            if command.startswith(text):
                completion = command[len(text):]
                yield Completion(completion, display=command)
        
        for command in self.command_map:
            if command.startswith(text):
                completion = command[len(text):]
                yield Completion(completion, display=command)

class REPLClient:
    """A REPL interface for the NexusServer connection."""
    
    def __init__(self, host: str, port: int):
        self.connection = NexusConnection(host, port)
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup command history
        history_path = os.path.expanduser("~/.nexus_client_history")
        self.history = FileHistory(history_path)
        self.prompt = PromptSession[str](history=self.history)

        self.command_map: dict[str, CommandInfo] = {}

    async def initialize_commands(self):
        """
        Initialize the command map with commands from the server
        """
        commands = await self.connection.list_commands()
        for command in commands:
            self.command_map[command.name] = command
    
    async def start(self):
        """Start the REPL interface"""
        self.running = True
        show_connect_prompt = True
        
        print("NexusClient REPL")
        print("Type 'help' for available commands or 'exit' to quit")
        
        # Auto-connect on first run
        print("Automatically connecting to server...")
        await self.connection.connect()
        
        await self.initialize_commands()

        self.connection.subscribe("server_message", self._handle_server_message)
        
        with patch_stdout():
            while self.running:
                # Check if we're connected (for reconnection after a disconnect)
                if not self.connection.connected and show_connect_prompt:
                    try:
                        show_connect_prompt = False
                        choice = await self.prompt.prompt_async("Not connected to server. Connect now? (y/n): ")
                        choice = choice.strip().lower()
                        if choice == 'y':
                            if not await self.connection.connect():
                                continue
                        else:
                            print("You can still use local commands. Type 'connect' to connect later.")
                    except (KeyboardInterrupt, EOFError):
                        print("\nExiting...")
                        self.running = False
                        break
                
                # Get command input
                try:
                    cmd = await self._command_prompt()
                    cmd = cmd.strip()
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    print("\nExiting...")
                    self.running = False
                    break
                
                if not cmd:
                    continue
                    
                # Process basic commands
                cmd_parts = cmd.split()
                cmd_name = cmd_parts[0].lower()
                
                if cmd_name == 'exit' or cmd_name == 'quit':
                    print("Exiting...")
                    break
                    
                elif cmd_name == 'help':
                    self.print_help()
                    
                elif cmd_name == 'connect':
                    await self.connection.connect()
                    
                elif cmd_name == 'disconnect':
                    await self.connection.disconnect()
                    
                elif cmd_name == 'status':
                    print(f"Connected: {self.connection.connected}")
                    print(f"Server address: {self.connection.host}:{self.connection.port}")
                
                elif cmd_name == 'list':
                    self._list_commands()
                
                elif await self._process_server_command(cmd):
                    continue

                else:
                    print(f"Unknown command: {cmd_name}")
                    print("Type 'help' for available commands")
            
            # Clean up when exiting
            if self.connection.connected:
                await self.connection.disconnect()
    
    async def stop(self):
        self.running = False
        self.shutdown_event.set()  # Signal any pending input to exit

        if self.connection.connected:
            await self.connection.disconnect()
    
    async def _command_prompt(self):
        repl_commands = ["connect", "disconnect", "status", "help", "exit", "quit", "list"]
        return await self.prompt.prompt_async(
            "nexus> ",
            completer=CommandCompleter(repl_commands, self.command_map),
            auto_suggest=AutoSuggestFromHistory()
        )
    
    async def _process_server_command(self, input: str):
        """
        Process the input string as a potential server command.
        
        Args:
            input (str): The input string to process
        
        Returns:
            bool: True if the command was processed, False otherwise
        """
        parts = shlex.split(input)
        cmd_name = parts[0].lower()

        if cmd_name in self.command_map:
            command_info = self.command_map[cmd_name]
            # Index parameters by position
            expected_params = {i + 1: param_name for i, param_name in enumerate(command_info.parameters)}
            
            # Parse arguments
            args = {}
            for i in range(1, len(parts)):
                if i in expected_params:
                    param_name = expected_params[i]
                    # TODO: Validate arguments
                    args[param_name] = parts[i]
                else:
                    print(f"Invalid argument: {parts[i]}")
                    return True

            await self._execute_command(command_info, args)
            return True
        
        return False

    async def _execute_command(self, command_info: CommandInfo, args: dict):
        """
        Execute a command with the given arguments
        """
        try:
            response = await self.connection.send_command(command_info.name, args)
            print(response)
        except Exception as e:
            print(f"Error executing command {command_info.name}: {e}")
 
    def _list_commands(self):
        """List all available server commands"""
        print("\nAvailable Server Commands:")
        if not self.command_map:
            print("  No commands available. Server may not be connected.")
        else:
            columns = []
            for name, cmd_info in sorted(self.command_map.items()):
                params = ", ".join(cmd_info.parameters.keys()) if cmd_info.parameters else ""
                if params:
                    params = f"[{params}]"
                columns.append((name, params, cmd_info.description))

            if columns:
                column_widths = [max(len(col[i]) for col in columns) for i in range(len(columns[0]))]
                for name, params, description in columns:
                    print(f"  {name:<{column_widths[0]}} {params:<{column_widths[1]}} - {description:<{column_widths[2]}}")
    
        print()

    def _handle_server_message(self, message: str):
        print(f"\r[Server message] {message}")
    
    def print_help(self):
        """Print help information"""
        print("\nClient Commands:")
        print("  help           - Show this help message")
        print("  list           - List all available server commands")
        print("  connect        - Connect to the server")
        print("  disconnect     - Disconnect from the server")
        print("  status         - Show connection status")
        print("  exit, quit     - Exit the client")
        
        # List server commands
        if self.command_map:
            print("\nServer Commands:")
            for name, cmd_info in sorted(self.command_map.items()):
                params = ", ".join(cmd_info.parameters.keys()) if cmd_info.parameters else ""
                if params:
                    print(f"  {name} [{params}] - {cmd_info.description}")
                else:
                    print(f"  {name} - {cmd_info.description}")
        print()
            
async def run_repl(args: argparse.Namespace):
    # Give some time for logfire to initialize
    await asyncio.sleep(.2)

    # Create and start client
    repl = REPLClient(args.host, args.port)
    try:
        # Start REPL - this will handle both user input and server messages concurrently
        await repl.start()
    finally:
        # Ensure we disconnect properly when exiting
        if repl.connection.connected:
            await repl.connection.disconnect()

# Synchronous entry point for command-line execution
def main():
    """Synchronous entry point that runs the async main function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="NexusVoice Test Client")
        parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
        parser.add_argument("--port", type=int, default=8008, help="Server port (default: 8008)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.TRACE) # type: ignore
            
            # TODO: redo bootstrap process so we don't have to dig deep into 
            # framework internals to reconfigure
            instance = get_logfire()
            assert instance is not None
            assert isinstance(instance.config.console, logfire.ConsoleOptions)
            instance.config.console.min_log_level = "trace"
            instance.config._initialized = False
            instance.config.initialize()

        
        asyncio.run(run_repl(args))
    except KeyboardInterrupt:
        print("\nClient terminated by user")
    return 0

if __name__ == "__main__":
    main()
